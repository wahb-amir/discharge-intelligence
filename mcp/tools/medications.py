from fhir.client import fhir_get
from mcp.tools._utils import truncate_text


async def get_patient_medications(patient_id: str, fhir_token: str) -> dict:
    """
    Fetches medications for a patient.
    Returns a compact list for reconciliation.
    """
    try:
        data = await fhir_get(
            f"MedicationRequest?patient={patient_id}&_count=20",
            fhir_token
        )

        medications = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                med = {
                    "name": truncate_text(
                        resource.get("medicationCodeableConcept", {}).get("text", "Unknown"),
                        120
                    ),
                    "status": resource.get("status", "unknown"),
                    "dosage": truncate_text(
                        resource.get("dosageInstruction", [{}])[0].get("text", "No dosage info"),
                        140
                    ),
                    "intent": resource.get("intent", "unknown")
                }
                medications.append(med)

        # active first, then cap total
        medications = sorted(medications, key=lambda x: x["status"] != "active")
        medications = medications[:10]

        return {
            "patient_id": patient_id,
            "medications": medications,
            "total_returned": len(medications)
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "medications": [],
            "error": str(e)
        }