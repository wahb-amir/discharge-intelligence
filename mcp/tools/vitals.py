from fhir.client import fhir_get
from mcp.tools._utils import truncate_text, take_first_unique


async def get_patient_vitals(patient_id: str, fhir_token: str) -> dict:
    try:
        data = await fhir_get(
            f"Observation?patient={patient_id}&category=vital-signs&_count=20&_sort=-date",
            fhir_token
        )

        vitals = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                value_quantity = resource.get("valueQuantity", {})

                vital = {
                    "name": truncate_text(
                        resource.get("code", {}).get("text", "Unknown"),
                        120
                    ),
                    "value": value_quantity.get("value", resource.get("valueString", "N/A")),
                    "unit": truncate_text(value_quantity.get("unit", ""), 20),
                    "date": resource.get("effectiveDateTime", "unknown")
                }
                vitals.append(vital)

        # keep latest unique vital names only, cap total
        vitals = take_first_unique(vitals, key="name", limit=8)

        return {
            "patient_id": patient_id,
            "vitals": vitals,
            "total_returned": len(vitals)
        }
    except Exception as e:
        return {"patient_id": patient_id, "vitals": [], "error": str(e)}