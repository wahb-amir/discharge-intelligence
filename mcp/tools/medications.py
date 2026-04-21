from fhir.client import fhir_get

async def get_patient_medications(
    patient_id: str,
    fhir_token: str
) -> dict:
    """
    Fetches all medication requests for a patient.
    Returns both active and stopped medications
    so reconciliation agent can compare them.
    """
    try:
        data = await fhir_get(
            f"MedicationRequest?patient={patient_id}&_count=50",
            fhir_token
        )
        
        medications = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                med = {
                    "name": resource.get("medicationCodeableConcept", {})
                                   .get("text", "Unknown"),
                    "status": resource.get("status", "unknown"),
                    "dosage": resource.get("dosageInstruction", [{}])[0]
                                     .get("text", "No dosage info"),
                    "intent": resource.get("intent", "unknown")
                }
                medications.append(med)
        
        return {
            "patient_id": patient_id,
            "medications": medications,
            "total": len(medications)
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "medications": [],
            "error": str(e)
        }