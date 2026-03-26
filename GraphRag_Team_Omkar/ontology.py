from enum import Enum

class NodeType(str, Enum):
    PATIENT = "Patient"
    ENCOUNTER = "Encounter"
    DIAGNOSIS = "Diagnosis"
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    DIAG_TEST = "DiagnosticTest"
    VITAL_SIGN = "VitalSign"
    CARE_PLAN = "CarePlan"
    ASSESSMENT = "Assessment"
    MEDICAL_HISTORY = "MedicalHistory"  # Intake Form Component
    SOCIAL_HISTORY = "SocialHistory"    # Intake Form Component
    BAYESIAN_NODE  = "BayesianNode"     # Bayesian Network variable node

RELATION_TYPES = [
    "HAS_PROFILE_DATA",
    "HAS_MEDICAL_HISTORY",
    "HAS_SOCIAL_HISTORY",
    "HAS_ENCOUNTER",
    "HAS_DIAGNOSIS",
    "HAS_SYMPTOM",
    "HAS_MEDICATION",
    "NEXT",           # Temporal Flow
    "CAUSED_BY",        # Causal logic for Chronic Conditions
    "HAS_FOLLOWUP",
    # ── Bayesian Network relations ──────────────────────────────────────────
    "BAYES_EDGE",       # Directed edge between two BayesianNode variables
    "HAS_PROBABILITY",  # BayesianNode → serialised CPT property
    "RISK_FACTOR_FOR",  # Semantic shorthand (e.g. Obesity RISK_FACTOR_FOR Diabetes)
]