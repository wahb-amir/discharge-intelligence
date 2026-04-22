from fhir.client import fhir_get
from mcp.tools._utils import truncate_text


async def get_patient_documents(patient_id: str, fhir_token: str) -> dict:
    try:
        data = await fhir_get(
            f"DocumentReference?patient={patient_id}&_count=5",
            fhir_token
        )

        documents = []
        if "entry" in data:
            for entry in data["entry"]:
                resource = entry.get("resource", {})
                content0 = resource.get("content", [{}])[0]
                attachment = content0.get("attachment", {})

                title = attachment.get("title") or resource.get("description") or "Unknown"

                doc = {
                    "type": truncate_text(
                        resource.get("type", {}).get("text", "Unknown"),
                        80
                    ),
                    "date": resource.get("date", "unknown"),
                    "status": resource.get("status", "unknown"),
                    "title": truncate_text(title, 120),
                }
                documents.append(doc)

        documents = documents[:5]

        return {
            "patient_id": patient_id,
            "documents": documents,
            "total_returned": len(documents)
        }
    except Exception as e:
        return {"patient_id": patient_id, "documents": [], "error": str(e)}