from fhir.client import fhir_get

async def get_patient_vitals(
    patient_id: str,
    fhir_token: str
) -> dict:
    try:
        data = await fhir_get(
            f"Observation?patient={patient_id}&category=vital-signs&_count=20&_sort=-date",
            fhir_token
        )
        
        vitals = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                vital = {
                    "name": resource.get("code", {})
                                   .get("text", "Unknown"),
                    "value": resource.get("valueQuantity", {})
                                    .get("value", "N/A"),
                    "unit": resource.get("valueQuantity", {})
                                   .get("unit", ""),
                    "date": resource.get("effectiveDateTime", "unknown")
                }
                vitals.append(vital)
        
        return {"patient_id": patient_id, "vitals": vitals}
    except Exception as e:
        return {"patient_id": patient_id, "vitals": [], "error": str(e)}