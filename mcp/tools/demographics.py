import logging
from fhir.client import fhir_get

# 🔥 simple module logger (better than print in production)
logger = logging.getLogger("fhir.demographics")


async def get_patient_demographics(
    patient_id: str,
    fhir_token: str
) -> dict:
    try:
        logger.info(f"[DEMOGRAPHICS] Fetching patient_id={patient_id}")

        data = await fhir_get(f"Patient/{patient_id}", fhir_token)

        # 🔍 log raw response safely
        logger.info(f"[DEMOGRAPHICS] Raw response type: {type(data)}")
        logger.debug(f"[DEMOGRAPHICS] Raw data: {str(data)[:500]}")

        # Handle error from fhir_get properly
        if isinstance(data, dict) and "error" in data:
            logger.error(f"[DEMOGRAPHICS] FHIR error: {data['error']}")
            return {
                "patient_id": patient_id,
                "error": data["error"],
                "raw": data.get("raw")
            }

        name = data.get("name", [{}])[0]
        full_name = " ".join(
            name.get("given", []) + [name.get("family", "")]
        ).strip()

        result = {
            "patient_id": patient_id,
            "name": full_name,
            "gender": data.get("gender", "unknown"),
            "birthDate": data.get("birthDate", "unknown"),
            "language": data.get("communication", [{}])[0]
                           .get("language", {})
                           .get("text", "English")
        }

        logger.info(f"[DEMOGRAPHICS] Success for {patient_id}")
        return result

    except Exception as e:
        logger.exception(f"[DEMOGRAPHICS] Crash for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}