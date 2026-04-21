from fhir.client import fhir_get

async def get_patient_documents(
    patient_id: str,
    fhir_token: str
) -> dict:
    try:
        data = await fhir_get(
            f"DocumentReference?patient={patient_id}&_count=10",
            fhir_token
        )
        
        documents = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                doc = {
                    "type": resource.get("type", {})
                                   .get("text", "Unknown"),
                    "date": resource.get("date", "unknown"),
                    "status": resource.get("status", "unknown"),
                    "description": resource.get("description", "")
                }
                documents.append(doc)
        
        return {"patient_id": patient_id, "documents": documents}
    except Exception as e:
        return {"patient_id": patient_id, "documents": [], "error": str(e)}