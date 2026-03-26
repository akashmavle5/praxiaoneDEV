import re
from config import BN_GLUCOSE_THRESHOLD, BN_BMI_THRESHOLD
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Standard US Medical Ranges Definition
RANGES = {
    "glucose_fasting": {"normal_max": 99, "prediabetes_max": 125, "diabetes_min": 126},
    "glucose_pp": {"normal_max": 139, "prediabetes_max": 199, "diabetes_min": 200}, # Postprandial (PP)
    "bmi": {"underweight_max": 18.4, "normal_max": 24.9, "overweight_max": 29.9, "obese_min": 30.0},
    "systolic_bp": {"normal_max": 119, "elevated_max": 129, "stage1_max": 139, "stage2_min": 140},
    "diastolic_bp": {"normal_max": 79, "stage1_max": 89, "stage2_min": 90},
    "cholesterol": {"normal_max": 199, "borderline_max": 239, "high_min": 240},
    "sleep": {"short_max": 6, "normal_min": 7, "normal_max": 9, "long_min": 10}
}

class PraxiaGraphRAG:
    def __init__(self, graph_builder, bn_engine=None):
        self.graph = graph_builder
        self.embedder = graph_builder.embedder
        self.bn_engine = bn_engine  # Optional; enriches context with risk scores

    def _parse_intent(self, query: str):
        """Extract filter intent from natural language query, handling exact numbers and clinical range keywords."""
        q = query.lower()
        intent = {}

        # 1. Glucose Filters (Fasting vs Postprandial/PP)
        if "fasting glucose" in q:
            if "normal" in q: intent["glucose_max"] = RANGES["glucose_fasting"]["normal_max"]
            elif "prediabetes" in q: 
                intent.update({"glucose_min": 100, "glucose_max": RANGES["glucose_fasting"]["prediabetes_max"]})
            elif any(w in q for w in ["high", "diabetes", "diabetic"]): 
                intent["glucose_min"] = RANGES["glucose_fasting"]["diabetes_min"]
        elif "glucose" in q:  # Defaults to PP (Oral Glucose Tolerance Test style) if 'fasting' is not specified
            if "normal" in q: intent["glucose_max"] = RANGES["glucose_pp"]["normal_max"]
            elif "prediabetes" in q: 
                intent.update({"glucose_min": 140, "glucose_max": RANGES["glucose_pp"]["prediabetes_max"]})
            elif any(w in q for w in ["high", "diabetes", "diabetic"]): 
                intent["glucose_min"] = RANGES["glucose_pp"]["diabetes_min"]
            else:
                # Numeric parsing fallback
                m = re.search(r'glucose\s*(above|over|greater than|>)\s*(\d+\.?\d*)', q)
                if m: intent["glucose_min"] = float(m.group(2))
                m = re.search(r'glucose\s*(below|under|less than|<)\s*(\d+\.?\d*)', q)
                if m: intent["glucose_max"] = float(m.group(2))

        # 2. BMI / Obesity Filters
        if "bmi" in q:
            if "normal" in q: intent["bmi_max"] = RANGES["bmi"]["normal_max"]
            elif "overweight" in q: 
                intent.update({"bmi_min": 25.0, "bmi_max": RANGES["bmi"]["overweight_max"]})
            elif "obese" in q or "obesity" in q: intent["bmi_min"] = RANGES["bmi"]["obese_min"]
            else:
                m = re.search(r'bmi\s*(above|over|greater than|>)\s*(\d+\.?\d*)', q)
                if m: intent["bmi_min"] = float(m.group(2))
                m = re.search(r'bmi\s*(below|under|less than|<)\s*(\d+\.?\d*)', q)
                if m: intent["bmi_max"] = float(m.group(2))

        # 3. Blood Pressure Filters (Hypertension Dataset)
        if "systolic" in q:
            if "normal" in q: intent["sys_bp_max"] = RANGES["systolic_bp"]["normal_max"]
            elif "elevated" in q: intent.update({"sys_bp_min": 120, "sys_bp_max": RANGES["systolic_bp"]["elevated_max"]})
            elif "stage 1" in q: intent.update({"sys_bp_min": 130, "sys_bp_max": RANGES["systolic_bp"]["stage1_max"]})
            elif "stage 2" in q or "high" in q: intent["sys_bp_min"] = RANGES["systolic_bp"]["stage2_min"]
        
        # 4. Cholesterol Filters
        if "cholesterol" in q:
            if "normal" in q: intent["chol_max"] = RANGES["cholesterol"]["normal_max"]
            elif "borderline" in q: intent.update({"chol_min": 200, "chol_max": RANGES["cholesterol"]["borderline_max"]})
            elif "high" in q: intent["chol_min"] = RANGES["cholesterol"]["high_min"]

        # 5. Age Filters
        if "age" in q:
            m = re.search(r'age\s*(above|over|older than|>)\s*(\d+)', q)
            if m: intent["age_min"] = int(m.group(2))
            m = re.search(r'age\s*(below|under|younger than|<)\s*(\d+)', q)
            if m: intent["age_max"] = int(m.group(2))

        # Outcome/diagnosis filter
        if any(w in q for w in ["diabetic", "has diabetes", "diagnosed", "positive outcome"]):
            intent["has_diagnosis"] = True

        return intent

    @staticmethod
    def _binarise_patient(row: dict) -> dict:
        """Convert a patient's numeric fields to categorical BN-compatible evidence states using Standard US Ranges."""
        evidence = {}
        
        # BMI Classification
        if row.get("bmi") is not None:
            bmi = row["bmi"]
            if bmi < 18.5: evidence["BMI_Category"] = "underweight"
            elif bmi < 25.0: evidence["BMI_Category"] = "normal"
            elif bmi < 30.0: evidence["BMI_Category"] = "overweight"
            else: evidence["BMI_Category"] = "obese"

        # Glucose Classification (Defaulting to Postprandial ranges generally used in random sampling)
        if row.get("glucose") is not None:
            glu = row["glucose"]
            if glu < 140: evidence["Glucose_Category"] = "normal"
            elif glu < 200: evidence["Glucose_Category"] = "prediabetes"
            else: evidence["Glucose_Category"] = "diabetes"

        # BP Classification (AHA guidelines)
        sys_bp, dia_bp = row.get("systolic_bp"), row.get("diastolic_bp")
        if sys_bp is not None and dia_bp is not None:
            if sys_bp < 120 and dia_bp < 80: evidence["BP_Category"] = "normal"
            elif sys_bp < 130 and dia_bp < 80: evidence["BP_Category"] = "elevated"
            elif sys_bp < 140 or dia_bp < 90: evidence["BP_Category"] = "stage_1"
            else: evidence["BP_Category"] = "stage_2"

        # Cholesterol Classification
        if row.get("cholesterol") is not None:
            chol = row["cholesterol"]
            if chol < 200: evidence["Cholesterol_Category"] = "normal"
            elif chol < 240: evidence["Cholesterol_Category"] = "borderline"
            else: evidence["Cholesterol_Category"] = "high"

        # Diagnose Flag
        diag = (row.get("diagnosis") or "").lower()
        if "hypertens" in diag:
            evidence["Hypertension"] = "present"
        elif "diabet" in diag:
            evidence["Diabetes"] = "present"
            
        return evidence

    def retrieve_context(self, query: str) -> str:
        try:
            intent = self._parse_intent(query)

            # Build WHERE clauses dynamically
            where_clauses = []
            params = {}

            if "bmi_min" in intent: where_clauses.append("p.bmi >= $bmi_min"); params["bmi_min"] = intent["bmi_min"]
            if "bmi_max" in intent: where_clauses.append("p.bmi <= $bmi_max"); params["bmi_max"] = intent["bmi_max"]
            if "glucose_min" in intent: where_clauses.append("maxGlucose >= $glucose_min"); params["glucose_min"] = intent["glucose_min"]
            if "glucose_max" in intent: where_clauses.append("maxGlucose <= $glucose_max"); params["glucose_max"] = intent["glucose_max"]
            if "sys_bp_min" in intent: where_clauses.append("maxSysBP >= $sys_bp_min"); params["sys_bp_min"] = intent["sys_bp_min"]
            if "sys_bp_max" in intent: where_clauses.append("maxSysBP <= $sys_bp_max"); params["sys_bp_max"] = intent["sys_bp_max"]
            if "chol_min" in intent: where_clauses.append("maxChol >= $chol_min"); params["chol_min"] = intent["chol_min"]
            if "chol_max" in intent: where_clauses.append("maxChol <= $chol_max"); params["chol_max"] = intent["chol_max"]
            if "age_min" in intent: where_clauses.append("p.age >= $age_min"); params["age_min"] = intent["age_min"]
            if "age_max" in intent: where_clauses.append("p.age <= $age_max"); params["age_max"] = intent["age_max"]
            if "has_diagnosis" in intent: where_clauses.append("diagnosisName IS NOT NULL")

            where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            with self.graph.driver.session() as session:
                # Updated Cypher to fetch additional medical properties (BP, Cholesterol, Sleep)
                cypher = f"""
                    MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_TEST]->(t:DiagnosticTest)
                    WITH p, 
                         max(t.glucose) AS maxGlucose, 
                         max(t.insulin) AS maxInsulin,
                         max(t.systolic_bp) AS maxSysBP,
                         max(t.diastolic_bp) AS maxDiaBP,
                         max(t.cholesterol) AS maxChol
                    OPTIONAL MATCH (p)-[:HAS_ENCOUNTER]->(e2:Encounter)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
                    WITH p, maxGlucose, maxInsulin, maxSysBP, maxDiaBP, maxChol, collect(DISTINCT d.name)[0] AS diagnosisName
                    {where_str}
                    RETURN DISTINCT
                        p.id AS patient_id,
                        p.age AS age,
                        p.bmi AS bmi,
                        maxGlucose AS glucose,
                        maxInsulin AS insulin,
                        maxSysBP AS systolic_bp,
                        maxDiaBP AS diastolic_bp,
                        maxChol AS cholesterol,
                        diagnosisName AS diagnosis
                    ORDER BY p.id
                    LIMIT 10
                """
                result = session.run(cypher, **params)
                records = list(result)

                if not records:
                    return (
                        f"No patients found matching your query.\n"
                        f"Parsed filters: {intent or 'None — try: BMI normal, fasting glucose prediabetes, stage 1 hypertension'}"
                    )

                context_lines = [f"Found {len(records)} unique patient(s) matching your query:\n"]
                for row in records:
                    diag = row["diagnosis"] or "No known condition"
                    
                    # Formatting data output to conditionally skip missing values gracefully
                    metrics = [
                        f"Age: {row['age']}",
                        f"BMI: {row['bmi']}" if row['bmi'] else None,
                        f"Glu: {row['glucose']}" if row['glucose'] else None,
                        f"BP: {row['systolic_bp']}/{row['diastolic_bp']}" if row['systolic_bp'] else None,
                        f"Chol: {row['cholesterol']}" if row['cholesterol'] else None
                    ]
                    metrics_str = " | ".join(filter(None, metrics))
                    
                    line = f"\u2022 Patient {row['patient_id']} | {metrics_str} | {diag}"

                    # ── BN Risk Annotation ───────────────────────────────────
                    if self.bn_engine:
                        try:
                            evidence = self._binarise_patient(dict(row))
                            risk = self.bn_engine.query_risk(evidence)
                            # Show top-2 risks above 40% probability
                            high_risk = sorted(
                                [(node, vals["present"]) for node, vals in risk.items()
                                 if vals.get("present", 0) > 0.4 and node not in evidence],
                                key=lambda x: -x[1]
                            )[:2]
                            if high_risk:
                                risk_str = ", ".join(f"{n}: {round(p*100)}%" for n, p in high_risk)
                                line += f"\n    → BN Risk Prediction: {risk_str}"
                        except Exception:
                            pass  # BN annotation is non-blocking

                    context_lines.append(line)

                return "\n".join(context_lines)

        except Exception as e:
            return f"GraphRAG retrieval error: {str(e)}"

# ─── Separate Function ────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)
def get_graph_context(query):
    cypher = """
        MATCH (n)-[r]->(m)
        WHERE toLower(n.name) CONTAINS toLower($search)
           OR toLower(m.name) CONTAINS toLower($search)
        RETURN n.name AS source, type(r) AS relation, m.name AS target
        LIMIT 20
    """
    with driver.session() as session:
        result = session.run(cypher, search=query)
        context = ""

        for row in result:
            context += f"{row['source']} {row['relation']} {row['target']}\n"

    return context

def query(self, question):
    return self.retrieve_context(question)    