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


async def fetch_all_patient_data(
    patient_id: str,
    fhir_token: str
) -> dict:
    """
    Fetch all FHIR data in parallel.
    No waiting for one call before starting the next.
    """
    results = await asyncio.gather(
        get_patient_medications(patient_id, fhir_token),
        get_patient_conditions(patient_id, fhir_token),
        get_patient_labs(patient_id, fhir_token),
        get_patient_vitals(patient_id, fhir_token),
        get_patient_demographics(patient_id, fhir_token),
        get_patient_documents(patient_id, fhir_token),
        return_exceptions=True
    )

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

    return {
        "medication":  results[0] if not isinstance(results[0], Exception) else {"status": "ERROR", "conflicts": [], "notes": str(results[0])},
        "clinical":    results[1] if not isinstance(results[1], Exception) else {"status": "ERROR", "concerns": [], "notes": str(results[1])},
        "followup":    results[2] if not isinstance(results[2], Exception) else {"status": "ERROR", "required_appointments": [], "notes": str(results[2])},
        "education":   results[3] if not isinstance(results[3], Exception) else {"status": "ERROR", "patient_instructions": "", "notes": str(results[3])}
    }


def determine_verdict(assessments: dict) -> tuple[str, str]:
    """
    Determines final discharge verdict and risk level
    based on all specialist assessments.
    """
    med = assessments["medication"]
    clinical = assessments["clinical"]
    followup = assessments["followup"]

    # Hard blockers — cannot discharge
    hard_blocked = (
        med.get("status") == "FLAGGED" or
        clinical.get("status") == "NOT_READY"
    )

    # Soft blockers — flag but can proceed with plan
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
    """
    Uses Groq to write a clean, structured
    discharge readiness report from all assessments.
    """

    med = assessments["medication"]
    clinical = assessments["clinical"]
    followup = assessments["followup"]
    education = assessments["education"]

    blocking_issues = []
    ready_items = []
    recommended_actions = []

    # Medication
    if med.get("status") == "FLAGGED":
        for conflict in med.get("conflicts", []):
            blocking_issues.append(f"Medication: {conflict}")
        recommended_actions.append(
            "Resolve medication conflicts with pharmacy before discharge"
        )
    else:
        ready_items.append("Medications reconciled — no conflicts found")

    # Clinical
    if clinical.get("status") == "NOT_READY":
        for concern in clinical.get("concerns", []):
            blocking_issues.append(f"Clinical: {concern}")
        recommended_actions.append(
            "Address outstanding clinical concerns before discharge"
        )
    else:
        ready_items.append("Clinical status acceptable for discharge")

    # Follow-up
    if followup.get("status") == "INCOMPLETE":
        blocking_issues.append(
            "Follow-up appointments not confirmed"
        )
        for appt in followup.get("required_appointments", []):
            recommended_actions.append(f"Book: {appt}")
    else:
        ready_items.append("Follow-up plan in place")

    # Education always completes
    ready_items.append("Patient discharge instructions generated")

    prompt = f"""
Write a professional discharge readiness report for {patient_name}.

VERDICT: {verdict}
RISK LEVEL: {risk_level}

BLOCKING ISSUES:
{chr(10).join(f"- {i}" for i in blocking_issues) or "- None"}

READY ITEMS:
{chr(10).join(f"- {i}" for i in ready_items)}

RECOMMENDED ACTIONS:
{chr(10).join(f"- {a}" for a in recommended_actions) or "- None"}

MEDICATION ASSESSMENT:
{med.get("notes", "")}

CLINICAL ASSESSMENT:
{clinical.get("notes", "")}

FOLLOW-UP PLAN:
{followup.get("notes", "")}

Write this as a concise, professional clinical report.
Start with the verdict prominently.
Be specific about what is blocking discharge and why.
End with clear next steps for the clinical team.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior hospitalist writing a discharge "
                    "readiness report. You are concise, direct, and "
                    "clinically precise. Your reports are read by busy "
                    "clinicians who need clear verdicts and specific "
                    "action items immediately."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1200
    )

    return response.choices[0].message.content


async def run_orchestrator(
    message: str,
    patient_id: str,
    fhir_token: str
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

    # Step 1 — fetch all data in parallel
    print(f"[Orchestrator] Fetching FHIR data for patient {patient_id}")
    patient_data = await fetch_all_patient_data(patient_id, fhir_token)

    patient_name = patient_data["demographics"].get("name", "Patient")
    print(f"[Orchestrator] Data fetched for {patient_name}")

    # Step 2 — run all specialist agents in parallel
    print("[Orchestrator] Running specialist agents in parallel...")
    assessments = await run_all_specialist_agents(
        patient_id, fhir_token, patient_data
    )
    print("[Orchestrator] All specialist agents completed")

    # Step 3 — determine verdict
    verdict, risk_level = determine_verdict(assessments)
    print(f"[Orchestrator] Verdict: {verdict} | Risk: {risk_level}")

    # Step 4 — synthesize final report
    report = await synthesize_report(
        assessments, verdict, risk_level, patient_name
    )

    # Append patient instructions at the end
    patient_instructions = assessments["education"].get(
        "patient_instructions", ""
    )

    final_output = f"""
{report}

---

PATIENT DISCHARGE INSTRUCTIONS:
{patient_instructions}
"""

    return final_output