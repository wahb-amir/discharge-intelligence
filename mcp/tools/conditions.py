from fhir.client import fhir_get

async def get_patient_conditions(
    patient_id: str,
    fhir_token: str
) -> dict:
    try:
        data = await fhir_get(
            f"Condition?patient={patient_id}&_count=50",
            fhir_token
        )
        
        conditions = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                condition = {
                    "name": resource.get("code", {})
                                   .get("text", "Unknown"),
                    "status": resource.get("clinicalStatus", {})
                                     .get("coding", [{}])[0]
                                     .get("code", "unknown"),
                    "severity": resource.get("severity", {})
                                       .get("text", "Not specified"),
                    "onset": resource.get("onsetDateTime", "unknown")
                }
                conditions.append(condition)
        
        return {"patient_id": patient_id, "conditions": conditions}
    except Exception as e:
        return {"patient_id": patient_id, "conditions": [], "error": str(e)}