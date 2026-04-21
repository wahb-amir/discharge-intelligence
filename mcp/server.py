from mcp.tools.medications import get_patient_medications
from mcp.tools.labs import get_patient_labs
from mcp.tools.conditions import get_patient_conditions
from mcp.tools.vitals import get_patient_vitals
from mcp.tools.demographics import get_patient_demographics
from mcp.tools.documents import get_patient_documents

# Tool registry — Prompt Opinion reads this
# to know what tools your MCP server exposes
MCP_TOOLS = [
    {
        "name": "get_patient_medications",
        "description": "Fetches all current and historical medications for a patient including dosage and status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id", "fhir_token"]
        }
    },
    {
        "name": "get_patient_labs",
        "description": "Fetches recent laboratory results for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id", "fhir_token"]
        }
    },
    {
        "name": "get_patient_conditions",
        "description": "Fetches all active and historical conditions/diagnoses for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id", "fhir_token"]
        }
    },
    {
        "name": "get_patient_vitals",
        "description": "Fetches recent vital signs for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id", "fhir_token"]
        }
    },
    {
        "name": "get_patient_demographics",
        "description": "Fetches patient demographics including name, age, gender, and language",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id", "fhir_token"]
        }
    },
    {
        "name": "get_patient_documents",
        "description": "Fetches clinical documents and notes for a patient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "fhir_token": {"type": "string"}
            },
            "required": ["patient_id", "fhir_token"]
        }
    }
]

# Tool dispatcher
async def call_tool(
    tool_name: str,
    patient_id: str,
    fhir_token: str
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
    
    return await tools[tool_name](patient_id, fhir_token)