from fhir.client import fhir_get

async def get_patient_labs(
    patient_id: str,
    fhir_token: str
) -> dict:
    try:
        data = await fhir_get(
            f"Observation?patient={patient_id}&category=laboratory&_count=20&_sort=-date",
            fhir_token
        )
        
        labs = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                lab = {
                    "name": resource.get("code", {})
                                   .get("text", "Unknown"),
                    "value": resource.get("valueQuantity", {})
                                    .get("value", "N/A"),
                    "unit": resource.get("valueQuantity", {})
                                   .get("unit", ""),
                    "status": resource.get("status", "unknown"),
                    "date": resource.get("effectiveDateTime", "unknown"),
                    "interpretation": resource.get("interpretation", [{}])[0]
                                             .get("text", "Normal")
                }
                labs.append(lab)
        
        return {"patient_id": patient_id, "labs": labs}
    except Exception as e:
        return {"patient_id": patient_id, "labs": [], "error": str(e)}