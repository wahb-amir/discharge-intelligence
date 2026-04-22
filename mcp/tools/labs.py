from fhir.client import fhir_get
from mcp.tools._utils import truncate_text, take_first_unique


async def get_patient_labs(patient_id: str, fhir_token: str) -> dict:
    try:
        data = await fhir_get(
            f"Observation?patient={patient_id}&category=laboratory&_count=20&_sort=-date",
            fhir_token
        )

        labs = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                value_quantity = resource.get("valueQuantity", {})
                interpretation = resource.get("interpretation", [{}])

                lab = {
                    "name": truncate_text(
                        resource.get("code", {}).get("text", "Unknown"),
                        120
                    ),
                    "value": value_quantity.get("value", resource.get("valueString", "N/A")),
                    "unit": truncate_text(value_quantity.get("unit", ""), 20),
                    "status": resource.get("status", "unknown"),
                    "date": resource.get("effectiveDateTime", "unknown"),
                    "interpretation": truncate_text(
                        interpretation[0].get("text", "Normal") if interpretation else "Normal",
                        80
                    ),
                }
                labs.append(lab)

        # keep latest unique lab names only, cap total
        labs = take_first_unique(labs, key="name", limit=10)

        return {
            "patient_id": patient_id,
            "labs": labs,
            "total_returned": len(labs)
        }
    except Exception as e:
        return {"patient_id": patient_id, "labs": [], "error": str(e)}