import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

async def run_clinical_agent(
    patient_id: str,
    fhir_token: str,
    conditions: dict,
    labs: dict,
    vitals: dict
) -> dict:
    """
    Assesses whether the patient is clinically
    ready for discharge based on labs, vitals,
    and whether their admitting diagnosis
    has been adequately treated.
    """

    conditions_list = "\n".join([
        f"- {c['name']} (status: {c['status']}, "
        f"severity: {c['severity']})"
        for c in conditions.get("conditions", [])
    ]) or "No conditions recorded"

    labs_list = "\n".join([
        f"- {l['name']}: {l['value']} {l['unit']} "
        f"({l['interpretation']}) on {l['date']}"
        for l in labs.get("labs", [])
    ]) or "No labs recorded"

    vitals_list = "\n".join([
        f"- {v['name']}: {v['value']} {v['unit']} on {v['date']}"
        for v in vitals.get("vitals", [])
    ]) or "No vitals recorded"

    prompt = f"""
You are reviewing a patient's clinical status to determine
if they are ready for discharge from the hospital.

ACTIVE CONDITIONS / DIAGNOSES:
{conditions_list}

RECENT LAB RESULTS:
{labs_list}

RECENT VITAL SIGNS:
{vitals_list}

Assess clinical discharge readiness. Consider:
1. Are vital signs within safe discharge thresholds?
2. Are critical lab values within acceptable ranges?
3. Has the primary admitting diagnosis been adequately treated?
4. Are there any active clinical concerns that require 
   continued inpatient monitoring?

Respond in this exact format:

CLINICAL STATUS: [READY or NOT_READY]

CONCERNS:
(List specific values or findings that concern you with 
clinical reasoning. Write "None" if no concerns.)

NORMAL FINDINGS:
(List values that are within acceptable range)

CLINICAL NOTES:
(Overall clinical assessment and reasoning)
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced hospitalist physician "
                    "reviewing patients for discharge readiness. "
                    "You are thorough, evidence-based, and specific "
                    "about which values concern you and why. "
                    "You prioritize patient safety above speed of discharge."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        tools=LLM_TOOLS,              
        tool_choice="auto", 
        temperature=0.1,
        max_tokens=1000
    )

    content = response.choices[0].message.content

    # Parse status
    status = "NOT_READY" if "NOT_READY" in content.upper() else "READY"

    # Extract concerns
    concerns = []
    normal_findings = []
    lines = content.split("\n")
    
    current_section = None
    for line in lines:
        if "CONCERNS:" in line:
            current_section = "concerns"
            continue
        elif "NORMAL FINDINGS:" in line:
            current_section = "normal"
            continue
        elif "CLINICAL NOTES:" in line:
            current_section = "notes"
            continue
        
        if current_section == "concerns" and line.strip().startswith("-"):
            concerns.append(line.strip("- ").strip())
        elif current_section == "normal" and line.strip().startswith("-"):
            normal_findings.append(line.strip("- ").strip())

    return {
        "status": status,
        "concerns": concerns,
        "normal_findings": normal_findings,
        "full_assessment": content,
        "notes": f"Reviewed {len(labs.get('labs', []))} labs, "
                 f"{len(vitals.get('vitals', []))} vitals"
    }