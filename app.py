import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from agents.orchestrator import run_orchestrator
from mcp.server import call_tool

load_dotenv()

app = FastAPI(title="Discharge Intelligence Network")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25")

MCP_TOOLS = [
    {
        "name": "get_patient_medications",
        "description": "Fetches all current and historical medications for a patient including dosage and status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"},
            },
            "required": ["patient_id", "fhir_token"],
        },
    },
    {
        "name": "get_patient_labs",
        "description": "Fetches recent laboratory results for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"},
            },
            "required": ["patient_id", "fhir_token"],
        },
    },
    {
        "name": "get_patient_conditions",
        "description": "Fetches all active and historical conditions/diagnoses for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"},
            },
            "required": ["patient_id", "fhir_token"],
        },
    },
    {
        "name": "get_patient_vitals",
        "description": "Fetches recent vital signs for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"},
            },
            "required": ["patient_id", "fhir_token"],
        },
    },
    {
        "name": "get_patient_demographics",
        "description": "Fetches patient demographics including name, age, gender, and language",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"},
            },
            "required": ["patient_id", "fhir_token"],
        },
    },
    {
        "name": "get_patient_documents",
        "description": "Fetches clinical documents and notes for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"},
            },
            "required": ["patient_id", "fhir_token"],
        },
    },
]

MCP_TOOL_NAMES = {tool["name"] for tool in MCP_TOOLS}

API_KEYS = {
    key for key in (
        os.getenv("AGENT_API_KEY"),
        os.getenv("AGENT_API_KEY_PROD"),
    )
    if key
}


def verify_api_key(x_api_key: str | None) -> None:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")


def jsonrpc_error(request_id, code: int, message: str, data: dict | None = None):
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


def tool_result_to_text(result) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str, indent=2)


# ─── Agent Card ───────────────────────────────────────────
@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "Discharge Readiness Orchestrator",
        "description": (
            "Performs a comprehensive discharge readiness assessment. "
            "Call this when a clinician asks if a patient is ready for "
            "discharge, needs a discharge assessment, or wants to know "
            "what is blocking a patient's discharge."
        ),
        "version": "1.0.0",
        "url": os.getenv("PUBLIC_URL", "http://localhost:8000"),
        "skills": [
            {
                "id": "assess_discharge_readiness",
                "name": "Assess Discharge Readiness",
                "description": (
                    "Evaluates medication reconciliation, clinical status, "
                    "follow-up planning, and patient education to produce "
                    "a structured discharge readiness report with a clear "
                    "verdict and any blocking issues."
                ),
            }
        ],
    }


# ─── MCP Endpoints ────────────────────────────────────────
@app.get("/mcp")
def mcp_info(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    verify_api_key(x_api_key)
    return {
        "name": "Discharge Intelligence MCP",
        "version": "1.0.0",
        "tools": MCP_TOOLS,
    }


@app.post("/mcp")
async def mcp_post(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    print(f"[MCP] Received: {body}")

    method = body.get("method", "")
    request_id = body.get("id")

    # ─── Initialize handshake ─────────────────────────────
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": body.get("params", {}).get("protocolVersion", "2025-11-25"),
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    },
                    "extensions": {
                        "ai.promptopinion/fhir-context": {
                            "scopes": [
                                {
                                    "name": "patient/Patient.rs",
                                    "required": True
                                },
                                {
                                    "name": "patient/Condition.rs"
                                },
                                {
                                    "name": "patient/Observation.rs"
                                },
                                {
                                    "name": "patient/MedicationRequest.rs"
                                }
                            ]
                        },
                        "io.modelcontextprotocol/ui": {
                            "mimeTypes": ["text/html;profile=mcp-app"]
                        }
                    }
                },
                "serverInfo": {
                    "name": "Discharge Intelligence MCP",
                    "version": "1.0.0"
                }
            }
        }

    # ─── Initialized notification ─────────────────────────
    if method == "notifications/initialized":
        return Response(status_code=204)

    # ─── Tool discovery ───────────────────────────────────
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": MCP_TOOLS,
            },
        }

    # ─── Tool call ────────────────────────────────────────
    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in MCP_TOOL_NAMES:
            return jsonrpc_error(
                request_id,
                -32602,
                f"Unknown tool: {tool_name}",
            )

        patient_id = arguments.get("patient_id", "")
        fhir_token = arguments.get("fhir_token", "")

        if not patient_id or not fhir_token:
            return jsonrpc_error(
                request_id,
                -32602,
                "patient_id and fhir_token are required",
            )

        result = await call_tool(tool_name, patient_id, fhir_token)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": tool_result_to_text(result),
                    }
                ],
                "isError": False,
            },
        }

    # ─── Ping ─────────────────────────────────────────────
    if method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {},
        }

    # ─── Fallback ─────────────────────────────────────────
    print(f"[MCP] Unhandled method: {method}")
    return jsonrpc_error(
        request_id,
        -32601,
        f"Unhandled method: {method}",
    )


@app.post("/mcp/tools/{tool_name}")
async def mcp_tool_call(
    tool_name: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)

    if tool_name not in MCP_TOOL_NAMES:
        raise HTTPException(status_code=404, detail="Unknown tool")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    patient_id = body.get("patient_id")
    fhir_token = body.get("fhir_token")

    if not patient_id or not fhir_token:
        raise HTTPException(
            status_code=400,
            detail="patient_id and fhir_token required",
        )

    result = await call_tool(tool_name, patient_id, fhir_token)
    return {
        "tool": tool_name,
        "result": result,
    }


# ─── Agent Endpoints ──────────────────────────────────────
@app.post("/agents/orchestrator")
async def orchestrator_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = body.get("message", "")
    context = body.get("context", {})
    patient_id = context.get("patient_id", "")
    fhir_token = context.get("fhir_token", "")

    result = await run_orchestrator(
        message=message,
        patient_id=patient_id,
        fhir_token=fhir_token,
    )

    return {"response": result}


# ─── Health Check ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "Discharge Intelligence Network",
    }