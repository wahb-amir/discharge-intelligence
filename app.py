import os
import asyncio
from typing import Optional
from fastapi import FastAPI, Header, Header, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mcp.server import MCP_TOOLS, call_tool
from agents.orchestrator import run_orchestrator

load_dotenv()

app = FastAPI(title="Discharge Intelligence Network")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─── Agent Card ───────────────────────────────────────────
# Prompt Opinion reads this to know what your agent does
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
                )
            }
        ]
    }

# ─── MCP Endpoints ────────────────────────────────────────
@app.get("/mcp")
def mcp_info(x_api_key: Optional[str] = Header(None)):
    # Accept any key for now, just needs to not crash
    return {
        "name": "Discharge Intelligence MCP",
        "version": "1.0.0",
        "tools": MCP_TOOLS
    }

@app.post("/mcp/tools/{tool_name}")
async def mcp_tool_call(tool_name: str, request: Request):
    body = await request.json()
    patient_id = body.get("patient_id")
    fhir_token = body.get("fhir_token")
    
    if not patient_id or not fhir_token:
        raise HTTPException(
            status_code=400,
            detail="patient_id and fhir_token required"
        )
    
    result = await call_tool(tool_name, patient_id, fhir_token)
    return result

# ─── Agent Endpoints ──────────────────────────────────────
@app.post("/agents/orchestrator")
async def orchestrator_endpoint(request: Request):
    body = await request.json()
    
    # Extract context Prompt Opinion passes automatically
    message = body.get("message", "")
    context = body.get("context", {})
    patient_id = context.get("patient_id", "")
    fhir_token = context.get("fhir_token", "")
    
    result = await run_orchestrator(
        message=message,
        patient_id=patient_id,
        fhir_token=fhir_token
    )
    
    return {"response": result}

# ─── Health Check ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "running", "service": "Discharge Intelligence Network"}