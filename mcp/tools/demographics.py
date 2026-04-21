from fhir.client import fhir_get

async def get_patient_demographics(
    patient_id: str,
    fhir_token: str
) -> dict:
    try:
        data = await fhir_get(f"Patient/{patient_id}", fhir_token)
        
        name = data.get("name", [{}])[0]
        full_name = " ".join(
            name.get("given", []) + [name.get("family", "")]
        ).strip()
        
        return {
            "patient_id": patient_id,
            "name": full_name,
            "gender": data.get("gender", "unknown"),
            "birthDate": data.get("birthDate", "unknown"),
            "language": data.get("communication", [{}])[0]
                           .get("language", {})
                           .get("text", "English")
        }
    except Exception as e:
        return {"patient_id": patient_id, "error": str(e)}