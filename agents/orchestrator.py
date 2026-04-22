import os
import asyncio
from typing import Any, Dict

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

MAX_LIST_ITEMS = 8
MAX_TEXT_CHARS = 700
MAX_SYNTHESIS_PROMPT_CHARS = 12000
MAX_FINAL_INSTRUCTIONS_CHARS = 1200


def clip_text(text: Any, limit: int = MAX_TEXT_CHARS) -> str:
    if text is None:
        return "None"
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[TRUNCATED]"


def safe_trim_list(items: Any, limit: int = MAX_LIST_ITEMS) -> list:
    if not isinstance(items, list) or not items:
        return []
    return items[:limit]


def compact_assessment(a: dict) -> dict:
    if not isinstance(a, dict):
        return {
            "status": "ERROR",
            "conflicts": [],
            "concerns": [],
            "required_appointments": [],
            "notes": "Invalid assessment payload",
            "summary": "Invalid assessment payload",
        }

    return {
        "status": a.get("status", "UNKNOWN"),
        "conflicts": safe_trim_list(a.get("conflicts", []), 5),
        "concerns": safe_trim_list(a.get("concerns", []), 5),
        "required_appointments": safe_trim_list(a.get("required_appointments", []), 5),
        "notes": clip_text(a.get("notes", ""), 600),
        "summary": clip_text(a.get("summary", ""), 600),
        "patient_instructions": clip_text(a.get("patient_instructions", ""), 600),
    }


def compact_patient_data_for_agents(data: dict) -> dict:
    return {
        "medications": {
            "medications": safe_trim_list(
                data.get("medications", {}).get("medications", []),
                MAX_LIST_ITEMS,
            )
        },
        "conditions": {
            "conditions": safe_trim_list(
                data.get("conditions", {}).get("conditions", []),
                MAX_LIST_ITEMS,
            )
        },
        "labs": {
            "labs": safe_trim_list(
                data.get("labs", {}).get("labs", []),
                MAX_LIST_ITEMS,
            )
        },
        "vitals": {
            "vitals": safe_trim_list(
                data.get("vitals", {}).get("vitals", []),
                MAX_LIST_ITEMS,
            )
        },
        "demographics": data.get("demographics", {}) or {},
        "documents": {
            "documents": safe_trim_list(
                data.get("documents", {}).get("documents", []),
                3,
            )
        },
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
        return_exceptions=True,
    )

    fhir_keys = [
        "medications",
        "conditions",
        "labs",
        "vitals",
        "demographics",
        "documents",
    ]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[FHIR ERROR] {fhir_keys[i]}: {result}")

    return {
        "medications": results[0] if isinstance(results[0], dict) else {"medications": []},
        "conditions": results[1] if isinstance(results[1], dict) else {"conditions": []},
        "labs": results[2] if isinstance(results[2], dict) else {"labs": []},
        "vitals": results[3] if isinstance(results[3], dict) else {"vitals": []},
        "demographics": results[4] if isinstance(results[4], dict) else {},
        "documents": results[5] if isinstance(results[5], dict) else {"documents": []},
    }


async def run_all_specialist_agents(
    patient_id: str,
    fhir_token: str,
    data: dict
) -> dict:
    """
    Run all 4 specialist agents in parallel.
    Each gets compacted data to keep prompts small.
    """
    safe_data = compact_patient_data_for_agents(data)

    results = await asyncio.gather(
        run_medication_agent(
            patient_id,
            fhir_token,
            safe_data["medications"],
        ),
        run_clinical_agent(
            patient_id,
            fhir_token,
            safe_data["conditions"],
            safe_data["labs"],
            safe_data["vitals"],
        ),
        run_followup_agent(
            patient_id,
            fhir_token,
            safe_data["conditions"],
            safe_data["demographics"],
        ),
        run_education_agent(
            patient_id,
            fhir_token,
            safe_data["demographics"],
            safe_data["conditions"],
            safe_data["medications"],
        ),
        return_exceptions=True,
    )

    agent_keys = ["medication", "clinical", "followup", "education"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[AGENT ERROR] {agent_keys[i]}: {result}")

    return {
        "medication": results[0] if isinstance(results[0], dict) else {
            "status": "ERROR",
            "conflicts": [],
            "notes": str(results[0]),
            "summary": "Medication agent failed",
        },
        "clinical": results[1] if isinstance(results[1], dict) else {
            "status": "ERROR",
            "concerns": [],
            "notes": str(results[1]),
            "summary": "Clinical agent failed",
        },
        "followup": results[2] if isinstance(results[2], dict) else {
            "status": "ERROR",
            "required_appointments": [],
            "notes": str(results[2]),
            "summary": "Follow-up agent failed",
        },
        "education": results[3] if isinstance(results[3], dict) else {
            "status": "ERROR",
            "patient_instructions": "",
            "notes": str(results[3]),
            "summary": "Education agent failed",
        },
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
        med.get("status") == "FLAGGED"
        or clinical.get("status") == "NOT_READY"
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


def build_synthesis_prompt(
    assessments: dict,
    verdict: str,
    risk_level: str,
    patient_name: str
) -> str:
    med = compact_assessment(assessments["medication"])
    clinical = compact_assessment(assessments["clinical"])
    followup = compact_assessment(assessments["followup"])

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

Write a professional clinical report.
Start with the verdict prominently displayed.
Be specific about what is blocking discharge and why.
End with clear numbered next steps for the clinical team.
Keep it under 250 words.
""".strip()

    if len(prompt) > MAX_SYNTHESIS_PROMPT_CHARS:
        prompt = prompt[:MAX_SYNTHESIS_PROMPT_CHARS] + "\n\n...[TRUNCATED]"

    return prompt


async def synthesize_report(
    assessments: dict,
    verdict: str,
    risk_level: str,
    patient_name: str
) -> str:
    prompt = build_synthesis_prompt(assessments, verdict, risk_level, patient_name)

    print(f"[Orchestrator] synthesis prompt length: {len(prompt)} chars")

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior hospitalist writing a discharge readiness report. "
                    "You are concise, direct, and clinically precise. "
                    "Your reports are read by busy clinicians who need clear verdicts and specific action items immediately."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=600,
    )

    content = response.choices[0].message.content if response.choices else ""
    return content or "Unable to generate discharge report."


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
    if not patient_id or not str(fhir_token).strip():
        return (
            "Unable to perform discharge assessment: "
            "no patient context provided. Please select "
            "a patient before requesting a discharge assessment."
        )

    print(f"[Orchestrator] Starting assessment for patient {patient_id}")
    print(f"[Orchestrator] FHIR server: {fhir_server_url or 'using env default'}")

    patient_data = await fetch_all_patient_data(
        patient_id,
        fhir_token,
        fhir_server_url,
    )

    patient_name = patient_data.get("demographics", {}).get("name", "Patient")
    print(f"[Orchestrator] Data fetched for {patient_name}")
    print(f"[Orchestrator] Medications: {len(patient_data['medications'].get('medications', []))}")
    print(f"[Orchestrator] Conditions: {len(patient_data['conditions'].get('conditions', []))}")
    print(f"[Orchestrator] Labs: {len(patient_data['labs'].get('labs', []))}")
    print(f"[Orchestrator] Vitals: {len(patient_data['vitals'].get('vitals', []))}")
    print(f"[Orchestrator] Documents: {len(patient_data['documents'].get('documents', []))}")

    print("[Orchestrator] Running specialist agents in parallel...")
    assessments = await run_all_specialist_agents(
        patient_id,
        fhir_token,
        patient_data,
    )
    print("[Orchestrator] All specialist agents completed")
    print(f"[Orchestrator] Medication: {assessments['medication'].get('status')}")
    print(f"[Orchestrator] Clinical: {assessments['clinical'].get('status')}")
    print(f"[Orchestrator] Follow-up: {assessments['followup'].get('status')}")
    print(f"[Orchestrator] Education: {assessments['education'].get('status')}")

    verdict, risk_level = determine_verdict(assessments)
    print(f"[Orchestrator] Verdict: {verdict} | Risk: {risk_level}")

    report = await synthesize_report(
        assessments,
        verdict,
        risk_level,
        patient_name,
    )

    patient_instructions = clip_text(
        assessments["education"].get("patient_instructions", ""),
        MAX_FINAL_INSTRUCTIONS_CHARS,
    )

    final_output = f"""{report}

---

PATIENT DISCHARGE INSTRUCTIONS:
{patient_instructions}
"""

    return final_output