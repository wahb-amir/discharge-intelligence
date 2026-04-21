import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

async def run_education_agent(
    patient_id: str,
    fhir_token: str,
    demographics: dict,
    conditions: dict,
    medications: dict
) -> dict:
    """
    Generates plain-language discharge instructions
    the patient can actually understand.
    """

    # Build context from data other agents already fetched
    patient_name = demographics.get("name", "the patient")
    patient_language = demographics.get("language", "English")
    
    conditions_list = "\n".join([
        f"- {c['name']} ({c['status']})"
        for c in conditions.get("conditions", [])
    ]) or "No conditions recorded"

    medications_list = "\n".join([
        f"- {m['name']}: {m['dosage']} ({m['status']})"
        for m in medications.get("medications", [])
        if m['status'] == 'active'
    ]) or "No active medications"

    prompt = f"""
You are generating discharge instructions for a real patient.

Patient: {patient_name}
Language preference: {patient_language}

Active Conditions:
{conditions_list}

Current Medications:
{medications_list}

Write clear, compassionate discharge instructions this patient 
can actually understand at home. Use simple language, no medical 
jargon. Structure your response exactly as follows:

WHAT HAPPENED TO YOU:
(2-3 sentences explaining their condition in plain English)

YOUR MEDICATIONS:
(List each medication, what it's for, how to take it)

WHAT TO DO AT HOME:
(5-6 specific actionable instructions)

WARNING SIGNS - GO TO EMERGENCY IF:
(4-5 specific symptoms that require immediate care)

YOUR FOLLOW-UP:
(reminder to attend follow-up appointments)

Keep the tone warm and reassuring.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a patient education specialist at a hospital. "
                    "You write discharge instructions that patients can "
                    "actually understand and follow at home. You never use "
                    "medical jargon. You are warm, clear, and specific."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    instructions = response.choices[0].message.content

    return {
        "status": "COMPLETE",
        "patient_instructions": instructions,
        "notes": f"Instructions generated in {patient_language}"
    }