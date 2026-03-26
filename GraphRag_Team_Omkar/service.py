import logging
import time
from llm_extractor import extract_encounter_logic
from validator import validate_llm_json
from graph_builder import ChronicGraphBuilder
from bayesian_network import BayesianEngine
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graph_rag import get_graph_context
from query_router import classify_query

# Note: search_pdf function import kela naslyas check kara, 
# karan khali answer method madhe te vaparle aahe.

logger = logging.getLogger(__name__)

class PJKGService:

    def __init__(self):
        # Connects to your Neo4j Cloud instance
        self.graph = ChronicGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # Initialise Bayesian Engine — try to reload from Neo4j first
        saved_bn = self.graph.load_bayesian_network()

        if saved_bn:
            self.bn_engine = BayesianEngine.from_neo4j_dict(saved_bn)
            logger.info(
                "BayesianEngine: reloaded from Neo4j (%d records).",
                self.bn_engine._record_count
            )
        else:
            self.bn_engine = BayesianEngine()
            logger.info("BayesianEngine: initialised with seed DAG (no prior data).")

    def process_patient_encounter(self, patient_id, encounter_id, transcript, date):

        print(f"Processing encounter {encounter_id} for patient {patient_id}...")

        # 1. Extract medical entities using Med42
        raw_llm_output = extract_encounter_logic(transcript)

        # 2. Validate the JSON structure
        structured_data = validate_llm_json(raw_llm_output)

        # 3. Insert into the Neo4j PJKG with Retry Logic
        max_retries = 3

        for attempt in range(max_retries):
            try:
                self.graph.append_encounter(
                    patient_id=patient_id,
                    encounter_id=encounter_id,
                    date=date,
                    extracted_data=structured_data
                )

                # ── Dynamically retrain Bayesian Network ────────────────────────────
                try:
                    records = self.graph.get_all_patient_records()
                    n = self.bn_engine.update_from_patient_data(records)
                    self.graph.persist_bayesian_network(self.bn_engine)

                    logger.info(
                        "BN retrained on %d patient records and persisted.",
                        n
                    )

                except Exception as bn_err:
                    logger.warning("BN update skipped: %s", bn_err)

                return {
                    "status": "success",
                    "message": "Encounter added to Knowledge Graph"
                }

            except Exception as e:
                print(
                    f"Database insertion error: {e}. Retrying {attempt + 1}/{max_retries}..."
                )
                time.sleep(2)

        return {
            "status": "error",
            "message": "Failed to add encounter after multiple attempts."
        }

    def check_clinical_ranges(self, data: dict):
        from graph_rag import RANGES

        result = {}

        for key, value in data.items():
            if key == "glucose_fasting":
                if value <= RANGES["glucose_fasting"]["normal_max"]: result[key] = "NORMAL"
                elif value <= RANGES["glucose_fasting"]["prediabetes_max"]: result[key] = "PREDIABETES"
                else: result[key] = "DIABETES"
            elif key == "glucose_pp" or key == "glucose":
                if value <= RANGES["glucose_pp"]["normal_max"]: result[key] = "NORMAL"
                elif value <= RANGES["glucose_pp"]["prediabetes_max"]: result[key] = "PREDIABETES"
                else: result[key] = "DIABETES"
            elif key == "bmi":
                if value <= RANGES["bmi"]["underweight_max"]: result[key] = "UNDERWEIGHT"
                elif value <= RANGES["bmi"]["normal_max"]: result[key] = "NORMAL"
                elif value <= RANGES["bmi"]["overweight_max"]: result[key] = "OVERWEIGHT"
                else: result[key] = "OBESE"
            elif key == "systolic_bp":
                if value <= RANGES["systolic_bp"]["normal_max"]: result[key] = "NORMAL"
                elif value <= RANGES["systolic_bp"]["elevated_max"]: result[key] = "ELEVATED"
                elif value <= RANGES["systolic_bp"]["stage1_max"]: result[key] = "STAGE 1 HYPERTENSION"
                else: result[key] = "STAGE 2 HYPERTENSION"
            elif key == "diastolic_bp":
                if value <= RANGES["diastolic_bp"]["normal_max"]: result[key] = "NORMAL"
                elif value <= RANGES["diastolic_bp"]["stage1_max"]: result[key] = "STAGE 1 HYPERTENSION"
                else: result[key] = "STAGE 2 HYPERTENSION"
            elif key == "cholesterol":
                if value <= RANGES["cholesterol"]["normal_max"]: result[key] = "NORMAL"
                elif value <= RANGES["cholesterol"]["borderline_max"]: result[key] = "BORDERLINE"
                else: result[key] = "HIGH"
            elif key == "blood_pressure": # Old key fallback
                low, high = (60, 120)
                if value < low: result[key] = "LOW"
                elif value > high: result[key] = "HIGH"
                else: result[key] = "NORMAL"
            elif key == "insulin":
                low, high = (16, 166)
                if value < low: result[key] = "LOW"
                elif value > high: result[key] = "HIGH"
                else: result[key] = "NORMAL"
            elif key == "sleep":
                if value <= RANGES["sleep"]["short_max"]: result[key] = "SHORT"
                elif value <= RANGES["sleep"]["normal_max"]: result[key] = "NORMAL"
                else: result[key] = "LONG"

        return result

    def answer(self, question):

        intent = classify_query(question)

        print("ROUTING:", intent)

        # ---------------- FILE ----------------
        if intent == "FILE":

            file_context = search_pdf(question)

            return f"""
📄 FILE ANSWER
{file_context}
"""

        # ---------------- GRAPH ----------------
        elif intent == "GRAPH":

            graph_context = self.graph.retrieve_context(question)

            return f"""
🧠 GRAPH ANSWER
{graph_context}
"""

        # ---------------- AUTO ----------------
        else:

            return """
I could not determine the exact data source.

Try asking:
• about uploaded report
• about patient database
"""