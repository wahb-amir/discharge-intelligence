from pydantic import BaseModel
from typing import Optional

# What Prompt Opinion sends to your agents
class AgentRequest(BaseModel):
    message: str
    context: Optional[dict] = {}

# What your agents return
class AgentResponse(BaseModel):
    response: str
    status: str = "success"

# What each specialist agent returns to orchestrator
class MedicationAssessment(BaseModel):
    status: str          # "CLEAR" or "FLAGGED"
    conflicts: list[str]
    reconciled_medications: list[str]
    notes: str

class ClinicalAssessment(BaseModel):
    status: str          # "READY" or "NOT_READY"
    concerns: list[str]
    normal_findings: list[str]
    notes: str

class FollowUpAssessment(BaseModel):
    status: str          # "COMPLETE" or "INCOMPLETE"
    required_appointments: list[str]
    care_gaps: list[str]
    notes: str

class EducationAssessment(BaseModel):
    status: str          # "COMPLETE"
    patient_instructions: str
    notes: str

class DischargeReport(BaseModel):
    verdict: str                    # "READY" or "NOT_READY"
    risk_level: str                 # "LOW", "MEDIUM", "HIGH"
    blocking_issues: list[str]
    ready_items: list[str]
    recommended_actions: list[str]
    medication_summary: str
    clinical_summary: str
    followup_plan: str
    patient_instructions: str