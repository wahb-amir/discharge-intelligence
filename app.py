import json
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:7860")
A2A_PROTOCOL_BINDING = "https://a2a-protocol.org/bindings/http-json"

# Keep FHIR-context declarations enabled by default so PromptOpinion can
# display the consent UI and pass FHIR context into the agent.
REQUIRE_FHIR_CONTEXT = os.getenv("REQUIRE_FHIR_CONTEXT", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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


def build_agent_card() -> dict[str, Any]:
    return {
        "name": "Discharge Readiness Orchestrator",
        "description": (
            "Performs a comprehensive discharge readiness assessment. "
            "Call this when a clinician asks if a patient is ready for "
            "discharge, needs a discharge assessment, or wants to know "
            "what is blocking a patient's discharge."
        ),
        "version": "1.0.0",
        "url": PUBLIC_URL,
        "supportedInterfaces": [
            {
                "url": PUBLIC_URL,
                "protocolBinding": A2A_PROTOCOL_BINDING,
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
            "extensions": {
                "ai.promptopinion/fhir-context": {
                    "scopes": [
                        {
                            "name": "patient/Patient.rs",
                            "required": True,
                        },
                        {
                            "name": "patient/Condition.rs",
                        },
                        {
                            "name": "patient/MedicationRequest.rs",
                        },
                        {
                            "name": "patient/Observation.rs",
                        },
                        {
                            "name": "patient/DocumentReference.rs",
                        },
                    ]
                }
            },
        },
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
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
                "tags": ["discharge", "fhir", "clinical", "mcp"],
                "examples": [
                    "Is this patient ready for discharge?",
                    "What is blocking discharge for this patient?",
                ],
                "inputModes": ["application/json", "text/plain"],
                "outputModes": ["application/json", "text/plain"],
            }
        ],
    }


def extract_fhir_context_from_headers(
    x_patient_id: str | None,
    x_fhir_access_token: str | None,
    x_fhir_server_url: str | None,
    x_fhir_refresh_token: str | None,
    x_fhir_refresh_url: str | None,
) -> dict[str, str]:
    context: dict[str, str] = {}

    if x_patient_id:
        context["patient_id"] = x_patient_id
    if x_fhir_access_token:
        context["fhir_token"] = x_fhir_access_token
    if x_fhir_server_url:
        context["fhir_server_url"] = x_fhir_server_url
    if x_fhir_refresh_token:
        context["fhir_refresh_token"] = x_fhir_refresh_token
    if x_fhir_refresh_url:
        context["fhir_refresh_url"] = x_fhir_refresh_url

    return context


def build_mcp_initialize_result(requested_protocol_version: str | None) -> dict[str, Any]:
    negotiated_version = SUPPORTED_PROTOCOL_VERSION
    if requested_protocol_version == SUPPORTED_PROTOCOL_VERSION:
        negotiated_version = requested_protocol_version

    return {
        "protocolVersion": negotiated_version,
        "capabilities": {
            "tools": {
                "listChanged": False,
            },
            "extensions": {
                "ai.promptopinion/fhir-context": {
                    "scopes": [
                        {
                            "name": "patient/Patient.rs",
                            "required": True,
                        },
                        {
                            "name": "patient/Condition.rs",
                        },
                        {
                            "name": "patient/MedicationRequest.rs",
                        },
                        {
                            "name": "patient/Observation.rs",
                        },
                        {
                            "name": "patient/DocumentReference.rs",
                        },
                    ]
                },
                "io.modelcontextprotocol/ui": {
                    "mimeTypes": ["text/html;profile=mcp-app"],
                },
            },
        },
        "serverInfo": {
            "name": "Discharge Intelligence MCP",
            "version": "1.0.0",
        },
    }


# ─── Agent Card ───────────────────────────────────────────
@app.get("/.well-known/agent.json")
def agent_card():
    return build_agent_card()


@app.get("/.well-known/agent-card.json")
def agent_card_compat():
    return build_agent_card()


# ─── MCP Endpoints ────────────────────────────────────────
@app.get("/mcp")
def mcp_info():
    return {
        "name": "Discharge Intelligence MCP",
        "version": "1.0.0",
        "tools": MCP_TOOLS,
    }


@app.post("/mcp")
async def mcp_post(
    request: Request,
    x_patient_id: str | None = None,
    x_fhir_server_url: str | None = None,
    x_fhir_access_token: str | None = None,
    x_fhir_refresh_token: str | None = None,
    x_fhir_refresh_url: str | None = None,
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        return jsonrpc_error(None, -32600, "Invalid Request")

    print(f"[MCP] Received: {body}")

    method = body.get("method", "")
    request_id = body.get("id")
    params = body.get("params", {}) if isinstance(body.get("params", {}), dict) else {}

    # FHIR context can arrive either from PromptOpinion headers or from the JSON body.
    header_context = extract_fhir_context_from_headers(
        x_patient_id=x_patient_id,
        x_fhir_access_token=x_fhir_access_token,
        x_fhir_server_url=x_fhir_server_url,
        x_fhir_refresh_token=x_fhir_refresh_token,
        x_fhir_refresh_url=x_fhir_refresh_url,
    )

    # ─── Initialize handshake ─────────────────────────────
    if method == "initialize":
        requested_protocol_version = params.get("protocolVersion")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": build_mcp_initialize_result(requested_protocol_version),
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
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        if tool_name not in MCP_TOOL_NAMES:
            return jsonrpc_error(
                request_id,
                -32602,
                f"Unknown tool: {tool_name}",
            )

        patient_id = (
            arguments.get("patient_id")
            or header_context.get("patient_id")
            or ""
        )
        fhir_token = (
            arguments.get("fhir_token")
            or header_context.get("fhir_token")
            or ""
        )

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
    x_patient_id: str | None = None,
    x_fhir_server_url: str | None = None,
    x_fhir_access_token: str | None = None,
    x_fhir_refresh_token: str | None = None,
    x_fhir_refresh_url: str | None = None,
):
    if tool_name not in MCP_TOOL_NAMES:
        raise HTTPException(status_code=404, detail="Unknown tool")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    header_context = extract_fhir_context_from_headers(
        x_patient_id=x_patient_id,
        x_fhir_access_token=x_fhir_access_token,
        x_fhir_server_url=x_fhir_server_url,
        x_fhir_refresh_token=x_fhir_refresh_token,
        x_fhir_refresh_url=x_fhir_refresh_url,
    )

    patient_id = body.get("patient_id") or header_context.get("patient_id")
    fhir_token = body.get("fhir_token") or header_context.get("fhir_token")

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
async def orchestrator_endpoint(
    request: Request,
    x_patient_id: str | None = None,
    x_fhir_server_url: str | None = None,
    x_fhir_access_token: str | None = None,
    x_fhir_refresh_token: str | None = None,
    x_fhir_refresh_url: str | None = None,
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    body_context = body.get("context", {})
    if not isinstance(body_context, dict):
        body_context = {}

    header_context = extract_fhir_context_from_headers(
        x_patient_id=x_patient_id,
        x_fhir_access_token=x_fhir_access_token,
        x_fhir_server_url=x_fhir_server_url,
        x_fhir_refresh_token=x_fhir_refresh_token,
        x_fhir_refresh_url=x_fhir_refresh_url,
    )

    # Body context wins when present; headers fill any missing values.
    message = body.get("message", "")
    patient_id = body_context.get("patient_id") or header_context.get("patient_id") or ""
    fhir_token = body_context.get("fhir_token") or header_context.get("fhir_token") or ""

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


@app.get("/")
def root():
    return {
        "service": "Discharge Intelligence Network",
        "status": "running",
    }