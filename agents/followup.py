import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

async def run_followup_agent(
    patient_id: str,
    fhir_token: str,
    conditions: dict,
    demographics: dict
) -> dict:
    """
    Determines what follow-up appointments are
    clinically indicated based on the patient's
    conditions. Identifies care gaps.
    """

    conditions_list = "\n".join([
        f"- {c['name']} (status: {c['status']})"
        for c in conditions.get("conditions", [])
    ]) or "No conditions recorded"

    age = demographics.get("birthDate", "unknown")
    gender = demographics.get("gender", "unknown")

    prompt = f"""
You are a care coordinator planning post-discharge 
follow-up for a patient.

Patient: {gender}, DOB {age}

ACTIVE CONDITIONS:
{conditions_list}

Based on these conditions determine:
1. What specialist follow-up appointments are clinically 
   indicated and within what timeframe?
   (e.g., cardiology within 7 days for heart failure)
2. What primary care follow-up is needed?
3. What preventive care gaps exist based on age/gender?
   (overdue screenings, vaccinations, health checks)

Respond in this exact format:

FOLLOW-UP STATUS: [COMPLETE or INCOMPLETE]
(INCOMPLETE means critical follow-ups are missing or unconfirmed)

REQUIRED APPOINTMENTS:
(List each appointment type, specialty, and timeframe)
(e.g., "- Cardiology: within 7 days - HIGH PRIORITY")

CARE GAPS IDENTIFIED:
(List overdue preventive care items)
(Write "None identified" if none)

COORDINATION NOTES:
(Any additional follow-up planning notes)
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced care coordinator at a hospital. "
                    "You ensure patients have appropriate follow-up care "
                    "after discharge. You are specific about appointment "
                    "types, specialties, and timeframes based on clinical "
                    "guidelines. You also identify preventive care gaps."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=800
    )

    content = response.choices[0].message.content

    # Parse status
    status = "INCOMPLETE" if "INCOMPLETE" in content.upper() else "COMPLETE"

    # Extract appointments
    appointments = []
    care_gaps = []
    lines = content.split("\n")
    
    current_section = None
    for line in lines:
        if "REQUIRED APPOINTMENTS:" in line:
            current_section = "appointments"
            continue
        elif "CARE GAPS" in line:
            current_section = "gaps"
            continue
        elif "COORDINATION NOTES:" in line:
            current_section = "notes"
            continue

        if current_section == "appointments" and line.strip().startswith("-"):
            appointments.append(line.strip("- ").strip())
        elif current_section == "gaps" and line.strip().startswith("-"):
            care_gaps.append(line.strip("- ").strip())

    return {
        "status": status,
        "required_appointments": appointments,
        "care_gaps": care_gaps,
        "full_assessment": content,
        "notes": f"Follow-up plan generated for "
                 f"{len(conditions.get('conditions', []))} conditions"
    }