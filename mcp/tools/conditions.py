from fhir.client import fhir_get
from mcp.tools._utils import truncate_text


async def get_patient_conditions(patient_id: str, fhir_token: str) -> dict:
    try:
        data = await fhir_get(
            f"Condition?patient={patient_id}&_count=20",
            fhir_token
        )

        conditions = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                condition = {
                    "name": truncate_text(
                        resource.get("code", {}).get("text", "Unknown"),
                        120
                    ),
                    "status": resource.get("clinicalStatus", {}) \
                                     .get("coding", [{}])[0] \
                                     .get("code", "unknown"),
                    "severity": truncate_text(
                        resource.get("severity", {}).get("text", "Not specified"),
                        80
                    ),
                    "onset": resource.get("onsetDateTime", "unknown")
                }
                conditions.append(condition)

        # keep active first, then cap total
        conditions = sorted(conditions, key=lambda x: x["status"] != "active")
        conditions = conditions[:8]

        return {
            "patient_id": patient_id,
            "conditions": conditions,
            "total_returned": len(conditions)
        }
    except Exception as e:
        return {"patient_id": patient_id, "conditions": [], "error": str(e)}