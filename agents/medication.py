import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

async def run_medication_agent(
    patient_id: str,
    fhir_token: str,
    medications: dict
) -> dict:
    """
    Reconciles inpatient vs outpatient medications.
    Flags conflicts, duplicates, dangerous omissions.
    This is the wow moment in the demo.
    """

    med_list = medications.get("medications", [])

    if not med_list:
        return {
            "status": "CLEAR",
            "conflicts": [],
            "reconciled_medications": [],
            "notes": "No medications found for this patient"
        }

    # Separate active vs stopped for reconciliation
    active_meds = [m for m in med_list if m["status"] == "active"]
    stopped_meds = [m for m in med_list if m["status"] == "stopped"]

    active_formatted = "\n".join([
        f"- {m['name']}: {m['dosage']}"
        for m in active_meds
    ]) or "None"

    stopped_formatted = "\n".join([
        f"- {m['name']}: {m['dosage']}"
        for m in stopped_meds
    ]) or "None"

    prompt = f"""
You are performing medication reconciliation for a patient 
being discharged from the hospital.

CURRENT INPATIENT MEDICATIONS (active):
{active_formatted}

PREVIOUSLY STOPPED MEDICATIONS:
{stopped_formatted}

Perform a thorough medication reconciliation. Check for:
1. Dose discrepancies (same drug, different doses)
2. Duplicate medications (same drug class, different names)
3. Dangerous drug interactions
4. Medications that were stopped but should be restarted
5. Missing medications for known conditions

Respond in this exact format:

RECONCILIATION STATUS: [CLEAR or FLAGGED]

CONFLICTS FOUND:
(List each conflict with drug name, the issue, and clinical risk level)
(Write "None" if no conflicts)

RECONCILED MEDICATION LIST:
(Final recommended medication list with correct doses)

PHARMACIST NOTES:
(Any additional clinical notes or recommendations)
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical pharmacist specializing in "
                    "medication reconciliation at care transitions. "
                    "You are thorough, specific about drug names and "
                    "doses, and clear about clinical risk levels. "
                    "Patient safety is your top priority."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    content = response.choices[0].message.content

    # Determine status from response
    status = "FLAGGED" if "FLAGGED" in content.upper() else "CLEAR"

    # Extract conflicts as a list
    conflicts = []
    if status == "FLAGGED":
        lines = content.split("\n")
        in_conflicts = False
        for line in lines:
            if "CONFLICTS FOUND:" in line:
                in_conflicts = True
                continue
            if in_conflicts and line.strip().startswith("-"):
                conflicts.append(line.strip("- ").strip())
            if in_conflicts and line.strip() == "":
                in_conflicts = False

    # Build reconciled med list
    reconciled = [
        f"{m['name']}: {m['dosage']}"
        for m in active_meds
    ]

    return {
        "status": status,
        "conflicts": conflicts,
        "reconciled_medications": reconciled,
        "full_assessment": content,
        "notes": f"Reviewed {len(med_list)} medications, "
                 f"{len(active_meds)} active, "
                 f"{len(stopped_meds)} stopped"
    }