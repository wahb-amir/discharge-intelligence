import os
import asyncio
from groq import Groq
from dotenv import load_dotenv
from mcp.tools.medications import get_patient_medications
from mcp.tools.conditions import get_patient_conditions
from mcp.tools.labs import get_patient_labs
from mcp.tools.vitals import get_patient_vitals
from mcp.tools.demographics import get_patient_demographics
from mcp.tools.documents import get_patient_documents
from agents.medication import run_medication_agent
from agents.clinical import run_clinical_agent
from agents.followup import run_followup_agent
from agents.education import run_education_agent

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def summarize_list(items, keys, limit=5):
    """
    Convert raw tool output into compact LLM-safe text.
    """
    if not items:
        return "None"

    lines = []
    for item in items[:limit]:
        line = ", ".join(
            f"{k}: {item.get(k, 'N/A')}" for k in keys
        )
        lines.append(f"- {line}")

    return "\n".join(lines)
def clip_text(text, limit=800):
    if text is None:
        return "None"
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[TRUNCATED]"


def compact_list(items, limit=5):
    if not items:
        return "None"
    return "\n".join(f"- {clip_text(item, 180)}" for item in items[:limit])


def compact_assessment(a: dict) -> dict:
    return {
        "status": a.get("status", "UNKNOWN"),
        "conflicts": a.get("conflicts", [])[:5],
        "concerns": a.get("concerns", [])[:5],
        "required_appointments": a.get("required_appointments", [])[:5],
        "notes": clip_text(a.get("notes", ""), 600),
        "summary": clip_text(a.get("summary", ""), 600),
    }
    
async def fetch_all_patient_data(
    patient_id: str,
    fhir_token: str,
    fhir_server_url: str | None = None
) -> dict:
    """
    Fetch all FHIR data in parallel.
    Uses fhir_server_url from Prompt Opinion if provided,
    otherwise falls back to env variable.
    """
    # Override FHIR base URL if Prompt Opinion passed one
    if fhir_server_url:
        os.environ["FHIR_BASE_URL"] = fhir_server_url
        print(f"[Orchestrator] Using FHIR server: {fhir_server_url}")

    results = await asyncio.gather(
        get_patient_medications(patient_id, fhir_token),
        get_patient_conditions(patient_id, fhir_token),
        get_patient_labs(patient_id, fhir_token),
        get_patient_vitals(patient_id, fhir_token),
        get_patient_demographics(patient_id, fhir_token),
        get_patient_documents(patient_id, fhir_token),
        return_exceptions=True
    )

    # Log any FHIR errors so we can debug
    fhir_keys = ["medications", "conditions", "labs", "vitals", "demographics", "documents"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[FHIR ERROR] {fhir_keys[i]}: {result}")

    return {
        "medications": results[0] if not isinstance(results[0], Exception) else {"medications": []},
        "conditions":  results[1] if not isinstance(results[1], Exception) else {"conditions": []},
        "labs":        results[2] if not isinstance(results[2], Exception) else {"labs": []},
        "vitals":      results[3] if not isinstance(results[3], Exception) else {"vitals": []},
        "demographics":results[4] if not isinstance(results[4], Exception) else {},
        "documents":   results[5] if not isinstance(results[5], Exception) else {"documents": []}
    }


async def run_all_specialist_agents(
    patient_id: str,
    fhir_token: str,
    data: dict
) -> dict:
    """
    Run all 4 specialist agents in parallel.
    Each gets the data it needs from the
    already-fetched FHIR data.
    """
    results = await asyncio.gather(
        run_medication_agent(
            patient_id, fhir_token,
            data["medications"]
        ),
        run_clinical_agent(
            patient_id, fhir_token,
            data["conditions"],
            data["labs"],
            data["vitals"]
        ),
        run_followup_agent(
            patient_id, fhir_token,
            data["conditions"],
            data["demographics"]
        ),
        run_education_agent(
            patient_id, fhir_token,
            data["demographics"],
            data["conditions"],
            data["medications"]
        ),
        return_exceptions=True
    )

    # Log any agent errors
    agent_keys = ["medication", "clinical", "followup", "education"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[AGENT ERROR] {agent_keys[i]}: {result}")

    return {
        "medication": results[0] if not isinstance(results[0], Exception) else {"status": "ERROR", "conflicts": [], "notes": str(results[0])},
        "clinical":   results[1] if not isinstance(results[1], Exception) else {"status": "ERROR", "concerns": [], "notes": str(results[1])},
        "followup":   results[2] if not isinstance(results[2], Exception) else {"status": "ERROR", "required_appointments": [], "notes": str(results[2])},
        "education":  results[3] if not isinstance(results[3], Exception) else {"status": "ERROR", "patient_instructions": "", "notes": str(results[3])}
    }


def determine_verdict(assessments: dict) -> tuple[str, str]:
    """
    Determines final discharge verdict and risk level
    based on all specialist assessments.
    """
    med = assessments["medication"]
    clinical = assessments["clinical"]
    followup = assessments["followup"]

    hard_blocked = (
        med.get("status") == "FLAGGED" or
        clinical.get("status") == "NOT_READY"
    )

    soft_blocked = followup.get("status") == "INCOMPLETE"

    if hard_blocked:
        verdict = "NOT_READY"
        if med.get("status") == "FLAGGED" and clinical.get("status") == "NOT_READY":
            risk = "HIGH"
        else:
            risk = "MEDIUM"
    elif soft_blocked:
        verdict = "NOT_READY"
        risk = "MEDIUM"
    else:
        verdict = "READY"
        risk = "LOW"

    return verdict, risk


async def synthesize_report(
    assessments: dict,
    verdict: str,
    risk_level: str,
    patient_name: str
) -> str:
    med = compact_assessment(assessments["medication"])
    clinical = compact_assessment(assessments["clinical"])
    followup = compact_assessment(assessments["followup"])
    education = compact_assessment(assessments["education"])

    blocking_issues = []
    ready_items = []
    recommended_actions = []

    if med["status"] == "FLAGGED":
        blocking_issues.extend([f"Medication: {c}" for c in med["conflicts"]])
        recommended_actions.append("Resolve medication conflicts with pharmacy before discharge")
    else:
        ready_items.append("Medications reconciled — no conflicts found")

    if clinical["status"] == "NOT_READY":
        blocking_issues.extend([f"Clinical: {c}" for c in clinical["concerns"]])
        recommended_actions.append("Address outstanding clinical concerns before discharge")
    else:
        ready_items.append("Clinical status acceptable for discharge")

    if followup["status"] == "INCOMPLETE":
        blocking_issues.append("Follow-up appointments not confirmed")
        recommended_actions.extend([f"Book: {a}" for a in followup["required_appointments"]])
    else:
        ready_items.append("Follow-up plan in place")

    ready_items.append("Patient discharge instructions generated")

    prompt = f"""
        Write a concise discharge readiness report for {patient_name}.

        VERDICT: {verdict}
        RISK LEVEL: {risk_level}

        BLOCKING ISSUES:
        {chr(10).join(f"- {i}" for i in blocking_issues) or "- None"}

        READY ITEMS:
        {chr(10).join(f"- {i}" for i in ready_items) or "- None"}

        RECOMMENDED ACTIONS:
        {chr(10).join(f"- {a}" for a in recommended_actions) or "- None"}

        MEDICATION SUMMARY:
        Status: {med["status"]}
        Summary: {med["summary"]}
        Notes: {med["notes"]}

        CLINICAL SUMMARY:
        Status: {clinical["status"]}
        Summary: {clinical["summary"]}
        Notes: {clinical["notes"]}

        FOLLOW-UP SUMMARY:
        Status: {followup["status"]}
        Summary: {followup["summary"]}
        Notes: {followup["notes"]}

        Keep this under 250 words.
        Start with the verdict.
        End with numbered next steps.
        """

async def run_orchestrator(
    message: str,
    patient_id: str,
    fhir_token: str,
    fhir_server_url: str | None = None
) -> str:
    """
    Main orchestrator function.
    This is what Prompt Opinion calls via A2A.

    Flow:
    1. Fetch all FHIR data in parallel
    2. Run all specialist agents in parallel
    3. Determine verdict
    4. Synthesize final report
    """

    # Handle missing context gracefully
    if not patient_id or not fhir_token:
        return (
            "Unable to perform discharge assessment: "
            "no patient context provided. Please select "
            "a patient before requesting a discharge assessment."
        )

    print(f"[Orchestrator] Starting assessment for patient {patient_id}")
    print(f"[Orchestrator] FHIR server: {fhir_server_url or 'using env default'}")

    # Step 1 — fetch all FHIR data in parallel
    patient_data = await fetch_all_patient_data(
        patient_id, fhir_token, fhir_server_url
    )

    patient_name = patient_data["demographics"].get("name", "Patient")
    print(f"[Orchestrator] Data fetched for {patient_name}")
    print(f"[Orchestrator] Medications: {len(patient_data['medications'].get('medications', []))}")
    print(f"[Orchestrator] Conditions: {len(patient_data['conditions'].get('conditions', []))}")
    print(f"[Orchestrator] Labs: {len(patient_data['labs'].get('labs', []))}")
    print(f"[Orchestrator] Vitals: {len(patient_data['vitals'].get('vitals', []))}")

    # Step 2 — run all specialist agents in parallel
    print("[Orchestrator] Running specialist agents in parallel...")
    assessments = await run_all_specialist_agents(
        patient_id, fhir_token, patient_data
    )
    print("[Orchestrator] All specialist agents completed")
    print(f"[Orchestrator] Medication: {assessments['medication'].get('status')}")
    print(f"[Orchestrator] Clinical: {assessments['clinical'].get('status')}")
    print(f"[Orchestrator] Follow-up: {assessments['followup'].get('status')}")
    print(f"[Orchestrator] Education: {assessments['education'].get('status')}")

    # Step 3 — determine verdict
    verdict, risk_level = determine_verdict(assessments)
    print(f"[Orchestrator] Verdict: {verdict} | Risk: {risk_level}")

    # Step 4 — synthesize final report
    report = await synthesize_report(
        assessments, verdict, risk_level, patient_name
    )

    # Append patient instructions
    patient_instructions = assessments["education"].get(
        "patient_instructions", ""
    )

    final_output = f"""{report}

---

PATIENT DISCHARGE INSTRUCTIONS:
{patient_instructions}
"""

    return final_output