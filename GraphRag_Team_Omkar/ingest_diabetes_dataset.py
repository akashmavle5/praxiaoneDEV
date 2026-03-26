import csv
import os
import pandas as pd
from neo4j import GraphDatabase
from embedding import Embedder
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class DiabetesIngester:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedder = Embedder()

    def run_ingestion(self, csv_filepath):
        with open(csv_filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            with self.driver.session() as session:
                for idx, row in enumerate(reader):
                    patient_id = f"P-DIAB-{idx+1}"
                    encounter_id = f"ENC-DIAB-{idx+1}"
                    
                    # Patient Demographics
                    age = int(row.get("Age", 0))
                    bmi = float(row.get("BMI", 0.0))
                    
                    session.run("""
                    MERGE (p:Patient {id: $pid})
                    SET p.age = $age, p.bmi = $bmi
                    """, pid=patient_id, age=age, bmi=bmi)
                    
                    # Encounter
                    session.run("""
                    MATCH (p:Patient {id: $pid})
                    CREATE (e:Encounter {id: $eid, date: '2025-01-01'})
                    CREATE (p)-[:HAS_ENCOUNTER]->(e)
                    """, pid=patient_id, eid=encounter_id)
                    
                    # Diagnostics/Vitals
                    glucose = float(row.get("Glucose", 0))
                    bp = float(row.get("BloodPressure", 0))
                    skin = float(row.get("SkinThickness", 0))
                    insulin = float(row.get("Insulin", 0))
                    dpf = float(row.get("DiabetesPedigreeFunction", 0))
                    pregnancies = int(row.get("Pregnancies", 0))
                    
                    session.run("""
                    MATCH (e:Encounter {id: $eid})
                    CREATE (v:VitalSign {blood_pressure: $bp, bmi: $bmi})
                    CREATE (t:DiagnosticTest {glucose: $glucose, insulin: $insulin, skin_thickness: $skin, pedigree_function: $dpf})
                    CREATE (m:MedicalHistory {pregnancies: $pregnancies})
                    CREATE (e)-[:HAS_VITALS]->(v)
                    CREATE (e)-[:HAS_TEST]->(t)
                    CREATE (e)-[:HAS_MEDICAL_HISTORY]->(m)
                    """, eid=encounter_id, bp=bp, bmi=bmi, glucose=glucose, insulin=insulin, skin=skin, dpf=dpf, pregnancies=pregnancies)
                    
                    # Outcome
                    if row.get("Outcome") == "1":
                        emb = self.embedder.encode("Diabetes")
                        session.run("""
                        MATCH (e:Encounter {id: $eid})
                        CREATE (d:Diagnosis {name: 'Diabetes', icd10: 'E11.9', embedding: $emb})
                        CREATE (e)-[:HAS_DIAGNOSIS]->(d)
                        """, eid=encounter_id, emb=emb)
                    
                    if idx % 50 == 0:
                        print(f"Ingested {idx} rows...")

        print("Finished ingesting diabetes dataset.")
        self.driver.close()

def ingest_heart_dataset(csv_filepath: str):
    print("Starting heart dataset ingestion...")
    df = pd.read_csv(csv_filepath)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        for i, row in df.iterrows():
            pid = f"P-HEART-{i}"

            # Patient node
            session.run("""
            MERGE (p:Patient {id:$pid})
            SET p.age=$age, p.sex=$sex
            """, {
                "pid": pid,
                "age": row["Age"],
                "sex": row["Sex"]
            })

            # Chest Pain
            if row["ChestPainType"] in ["ATA","ASY","NAP","TA"]:
                session.run("""
                MERGE (s:Symptom {name:"chest_pain"})
                MERGE (p:Patient {id:$pid})
                MERGE (p)-[:HAS_SYMPTOM]->(s)
                """, {"pid": pid})

            # High Blood Pressure
            if row["RestingBP"] > 130:
                session.run("""
                MERGE (s:Symptom {name:"high_bp"})
                MERGE (p:Patient {id:$pid})
                MERGE (p)-[:HAS_SYMPTOM]->(s)
                """, {"pid": pid})

            # High Cholesterol
            if row["Cholesterol"] > 240:
                session.run("""
                MERGE (s:Symptom {name:"high_cholesterol"})
                MERGE (p:Patient {id:$pid})
                MERGE (p)-[:HAS_SYMPTOM]->(s)
                """, {"pid": pid})

            # Diagnosis
            if row["HeartDisease"] == 1:
                session.run("""
                MATCH (p:Patient {id:$pid})
                MERGE (d:Diagnosis {name:"heart_disease"})
                MERGE (p)-[:HAS_DIAGNOSIS]->(d)
                """, {"pid": pid})
                
            if i % 50 == 0:
                print(f"Ingested {i} rows from heart dataset...")

    print("Finished ingesting heart dataset.")
    driver.close()

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    ingester = DiabetesIngester(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    ingester.run_ingestion(csv_path)
