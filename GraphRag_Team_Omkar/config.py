import os

# Neo4j Aura Cloud Credentials
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://73ee80d4.databases.neo4j.io") # Replace with your Aura URI
NEO4J_USER = os.getenv("NEO4J_USERNAME", "73ee80d4") # The default username generated for your Aura instance
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "8Dz7Z_WOOpyeUT9QZK01-QkbmSVTWtG8m_CHwbirUoI") # Your Aura password

# Med42 / LLM Credentials
MED42_API_KEY = os.getenv("MED42_API_KEY", "your-med42-key")
MED42_BASE_URL = os.getenv("MED42_BASE_URL", "http://localhost:8080/v1")

# ── Bayesian Network Config ───────────────────────────────────────────────────
# Dirichlet prior strength for Bayesian parameter estimation.
# Higher = more weight on the prior (uniform), lower = data dominates faster.
BN_DIRICHLET_PRIOR   = int(os.getenv("BN_DIRICHLET_PRIOR",  "10"))

# Clinical binarisation thresholds (used to convert continuous values to present/absent)
BN_GLUCOSE_THRESHOLD = float(os.getenv("BN_GLUCOSE_THRESHOLD", "140"))  # mg/dL
BN_BMI_THRESHOLD     = float(os.getenv("BN_BMI_THRESHOLD",     "30"))   # kg/m²
BN_AGE_THRESHOLD     = float(os.getenv("BN_AGE_THRESHOLD",     "50"))   # years