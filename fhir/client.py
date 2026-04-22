import logging
import os
import httpx

logger = logging.getLogger("fhir.client")


async def fhir_get(resource_path: str, fhir_token: str) -> dict:
    """
    Generic FHIR GET request.
    All tools use this to fetch patient data.
    """
    base_url = os.getenv("FHIR_BASE_URL")
    if not base_url:
        return {"error": "FHIR_BASE_URL is not set"}

    url = f"{base_url.rstrip('/')}/{resource_path.lstrip('/')}"

    headers = {
        "Authorization": f"Bearer {fhir_token}",
        "Accept": "application/fhir+json",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            raw_text = response.text

            logger.info(f"[FHIR] GET {url}")
            logger.info(f"[FHIR] STATUS {response.status_code}")
            logger.info(f"[FHIR] CONTENT-TYPE {response.headers.get('content-type')}")
            logger.debug(f"[FHIR] RAW {raw_text[:800]}")

            if response.status_code < 200 or response.status_code >= 300:
                return {
                    "error": f"FHIR request failed with status {response.status_code}",
                    "raw": raw_text[:800],
                }

            try:
                return response.json()
            except Exception as e:
                return {
                    "error": f"FHIR response was not valid JSON: {str(e)}",
                    "raw": raw_text[:800],
                }

    except httpx.RequestError as e:
        return {"error": f"FHIR request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected FHIR client error: {str(e)}"}