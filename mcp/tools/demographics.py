import logging
from datetime import datetime, date

from fhir.client import fhir_get
from mcp.tools._utils import safe_get, truncate_text

logger = logging.getLogger("fhir.demographics")


def _compute_age(birth_date_str: str):
    try:
        if not birth_date_str or birth_date_str == "unknown":
            return None
        birth = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return age
    except Exception:
        return None


async def get_patient_demographics(patient_id: str, fhir_token: str) -> dict:
    try:
        logger.info(f"[DEMOGRAPHICS] Fetching patient_id={patient_id}")
        data = await fhir_get(f"Patient/{patient_id}", fhir_token)

        if isinstance(data, dict) and "error" in data:
            logger.error(f"[DEMOGRAPHICS] FHIR error: {data['error']}")
            return {
                "patient_id": patient_id,
                "error": data["error"],
                "raw": data.get("raw"),
            }

        name = data.get("name", [{}])[0]
        given = name.get("given", [])
        family = name.get("family", "")
        full_name = " ".join([*given, family]).strip()

        birth_date = data.get("birthDate", "unknown")
        result = {
            "patient_id": patient_id,
            "name": truncate_text(full_name, 80) or "Unknown",
            "gender": data.get("gender", "unknown"),
            "birthDate": birth_date,
            "age": _compute_age(birth_date),
            "language": safe_get(data, "communication", 0, "language", "text", default="English"),
        }

        logger.info(f"[DEMOGRAPHICS] Success for {patient_id}")
        return result

    except Exception as e:
        logger.exception(f"[DEMOGRAPHICS] Crash for {patient_id}: {str(e)}")
        return {"patient_id": patient_id, "error": str(e)}