import httpx
import os

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL")

async def fhir_get(resource_path: str, fhir_token: str) -> dict:
    """
    Generic FHIR GET request.
    All tools use this to fetch patient data.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{FHIR_BASE_URL}/{resource_path}",
            headers={
                "Authorization": f"Bearer {fhir_token}",
                "Accept": "application/fhir+json"
            },
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()