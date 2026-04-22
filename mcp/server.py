from typing import Any, Dict, Optional

from mcp.tools.medications import get_patient_medications
from mcp.tools.labs import get_patient_labs
from mcp.tools.conditions import get_patient_conditions
from mcp.tools.vitals import get_patient_vitals
from mcp.tools.demographics import get_patient_demographics
from mcp.tools.documents import get_patient_documents

MCP_TOOLS = [
    {
        "name": "get_patient_medications",
        "description": "Get a compact medication list for reconciliation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_labs",
        "description": "Get recent key lab results in compact form",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_conditions",
        "description": "Get the patient’s key conditions in compact form",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_vitals",
        "description": "Get recent vital signs in compact form",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_demographics",
        "description": "Get patient demographics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_documents",
        "description": "Get a short list of clinical documents",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id"]
        }
    }
]


async def call_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    fhir_token: Optional[str] = None
) -> dict:
    tools = {
        "get_patient_medications": get_patient_medications,
        "get_patient_labs": get_patient_labs,
        "get_patient_conditions": get_patient_conditions,
        "get_patient_vitals": get_patient_vitals,
        "get_patient_demographics": get_patient_demographics,
        "get_patient_documents": get_patient_documents,
    }

    if tool_name not in tools:
        return {"error": f"Unknown tool: {tool_name}"}

    patient_id = arguments.get("patient_id")
    if not patient_id:
        return {"error": "Missing required argument: patient_id"}

    token = arguments.get("fhir_token") or fhir_token
    if not token:
        return {"error": "Missing required argument: fhir_token"}

    return await tools[tool_name](patient_id, token)