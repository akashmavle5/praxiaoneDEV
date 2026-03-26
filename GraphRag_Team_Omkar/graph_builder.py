import json
from neo4j import GraphDatabase
from embedding import Embedder

class ChronicGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedder = Embedder()
        self._init_vector_index()

    def _init_vector_index(self):
        # Creates vector index for GraphRAG hybrid retrieval
        query = """
        CREATE VECTOR INDEX node_embedding_index IF NOT EXISTS
        FOR (n:Diagnosis) ON (n.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
        """
        with self.driver.session() as session:
            session.run(query)

    def build_patient_profile(self, patient_id, demographic, med_history, soc_history):
        # Paper Logic: p -> HAS_MEDICAL_HISTORY -> M, p -> HAS_SOCIAL_HISTORY -> S
        query = """
        MERGE (p:Patient {id: $pid})
        SET p += $demo
        MERGE (m:MedicalHistory {patient_id: $pid}) SET m += $med
        MERGE (s:SocialHistory {patient_id: $pid}) SET s += $soc
        MERGE (p)-[:HAS_MEDICAL_HISTORY]->(m)
        MERGE (p)-[:HAS_SOCIAL_HISTORY]->(s)
        """
        with self.driver.session() as session:
            session.run(query, pid=patient_id, demo=demographic, med=med_history, soc=soc_history)

    def append_encounter(self, patient_id, encounter_id, date, extracted_data):
        with self.driver.session() as session:
            # 1. Create Encounter
            session.run("""
            MATCH (p:Patient {id: $pid})
            CREATE (e:Encounter {id: $eid, date: $date})
            CREATE (p)-[:HAS_ENCOUNTER]->(e)
            WITH p, e
            MATCH (p)-[:HAS_ENCOUNTER]->(prev:Encounter) WHERE prev.id <> e.id
            MERGE (prev)-[:NEXT]->(e) // Temporal linking
            """, pid=patient_id, eid=encounter_id, date=date)

            # 2. Add Diagnoses with Embeddings & Causal Links
            for diag in extracted_data.get("Diagnosis", []):
                embedding = self.embedder.encode(diag["name"])
                session.run("""
                MATCH (e:Encounter {id: $eid})
                CREATE (d:Diagnosis {name: $name, icd10: $icd, embedding: $emb})
                CREATE (e)-[:HAS_DIAGNOSIS]->(d)
                """, eid=encounter_id, name=diag["name"], icd=diag.get("icd10", ""), emb=embedding)
                
                # Link Chronic Causality (e.g., Hypertension CAUSED_BY Obesity)
                if "caused_by" in diag and diag["caused_by"]:
                    session.run("""
                    MATCH (d1:Diagnosis {name: $curr_name}), (d2:Diagnosis {name: $prev_name})
                    MERGE (d1)-[:CAUSED_BY]->(d2)
                    """, curr_name=diag["name"], prev_name=diag["caused_by"])

    # ── Bayesian Network persistence ──────────────────────────────────────────

    def persist_bayesian_network(self, bn_engine) -> None:
        """
        Upsert the BN DAG structure and CPTs into Neo4j as BayesianNode /
        BAYES_EDGE nodes and relationships.  CPT values are stored as a JSON
        string on the BAYES_EDGE or on the node itself.
        """
        data = bn_engine.to_neo4j_dict()

        with self.driver.session() as session:
            # 1. Upsert each BayesianNode (store record_count as metadata)
            all_nodes = list({n for edge in data["edges"] for n in (edge["from"], edge["to"])})
            for name in all_nodes:
                session.run(
                    "MERGE (b:BayesianNode {name: $name}) "
                    "SET b.record_count = $rc",
                    name=name,
                    rc=data["record_count"],
                )

            # 2. Upsert BAYES_EDGE relationships
            for edge in data["edges"]:
                session.run(
                    "MATCH (a:BayesianNode {name: $from_}), (b:BayesianNode {name: $to_}) "
                    "MERGE (a)-[:BAYES_EDGE]->(b)",
                    from_=edge["from"],
                    to_=edge["to"],
                )

            # 3. Store CPT as a JSON property on each BayesianNode
            for cpd in data["cpds"]:
                session.run(
                    "MATCH (b:BayesianNode {name: $name}) "
                    "SET b.cpt_parents     = $parents, "
                    "    b.cpt_values_json = $values_json",
                    name=cpd["node"],
                    parents=json.dumps(cpd["parents"]),
                    values_json=cpd["values_json"],
                )

    def load_bayesian_network(self) -> dict | None:
        """
        Load the previously persisted BN from Neo4j.
        Returns a dict compatible with BayesianEngine.from_neo4j_dict(),
        or None if no BN has been persisted yet.
        """
        with self.driver.session() as session:
            # Load edges
            edge_result = session.run(
                "MATCH (a:BayesianNode)-[:BAYES_EDGE]->(b:BayesianNode) "
                "RETURN a.name AS from_, b.name AS to_"
            )
            edges = [{"from": r["from_"], "to": r["to_"]} for r in edge_result]

            if not edges:
                return None  # Nothing persisted yet

            # Load CPDs
            cpd_result = session.run(
                "MATCH (b:BayesianNode) WHERE b.cpt_values_json IS NOT NULL "
                "RETURN b.name AS node, b.cpt_parents AS parents, "
                "       b.cpt_values_json AS values_json, b.record_count AS rc"
            )
            cpds = []
            record_count = 0
            for r in cpd_result:
                cpds.append({
                    "node":        r["node"],
                    "parents":     json.loads(r["parents"]) if r["parents"] else [],
                    "values_json": r["values_json"],
                })
                record_count = r["rc"] or 0

            return {"edges": edges, "cpds": cpds, "record_count": record_count}

    def get_all_patient_records(self) -> list[dict]:
        """
        Return a flat list of patient metric dicts for BN parameter estimation.
        Updated to support Hypertension, Obesity, and Depression datasets along with Diabetes.
        """
        cypher = """
        MATCH (p:Patient)
        OPTIONAL MATCH (p)-[:HAS_ENCOUNTER]->(:Encounter)-[:HAS_TEST]->(t:DiagnosticTest)
        WITH p, 
             max(t.glucose) AS glucose, 
             max(t.insulin) AS insulin,
             max(t.systolic_bp) AS systolic_bp,
             max(t.diastolic_bp) AS diastolic_bp,
             max(t.cholesterol) AS cholesterol,
             max(t.sleep_duration) AS sleep_duration,
             max(t.heart_rate) AS heart_rate
        OPTIONAL MATCH (p)-[:HAS_ENCOUNTER]->(:Encounter)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
        WITH p, glucose, insulin, systolic_bp, diastolic_bp, cholesterol, sleep_duration, heart_rate, collect(DISTINCT d.name)[0] AS diagnosis_name
        RETURN
            p.id        AS patient_id,
            p.age       AS age,
            p.bmi       AS bmi,
            p.smoking   AS smoking,
            p.inactive  AS inactive,
            glucose,
            insulin,
            systolic_bp,
            diastolic_bp,
            cholesterol,
            sleep_duration,
            heart_rate,
            diagnosis_name
        """
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(r) for r in result]
