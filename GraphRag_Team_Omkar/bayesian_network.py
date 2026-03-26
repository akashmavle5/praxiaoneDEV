# e:\internship_gemini\bayesian_network.py
"""
Dynamic Bayesian Network engine for Praxia5Chronic.

Uses pgmpy to maintain a clinically-grounded Bayesian Network (BN) whose
Conditional Probability Tables (CPTs) are re-estimated from real Neo4j patient
data after every new encounter. The DAG structure + CPTs are also persisted back
to Neo4j so the model survives server restarts.

All variables are *binary*: "present" or "absent".
"""

import json
import logging
import pandas as pd
from typing import Any

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

from config import BN_DIRICHLET_PRIOR, BN_GLUCOSE_THRESHOLD, BN_BMI_THRESHOLD, BN_AGE_THRESHOLD

logger = logging.getLogger(__name__)

# ── Clinically-grounded seed DAG ─────────────────────────────────────────────
# Each tuple is a directed edge: (cause, effect)
SEED_EDGES = [
    ("Obesity",             "Diabetes"),
    ("Obesity",             "Hypertension"),
    ("Obesity",             "Dyslipidemia"),
    ("Diabetes",            "ChronicKidneyDisease"),
    ("Diabetes",            "CardiovascularDisease"),
    ("Hypertension",        "ChronicKidneyDisease"),
    ("Hypertension",        "CardiovascularDisease"),
    ("ChronicKidneyDisease","Anemia"),
    ("Dyslipidemia",        "CardiovascularDisease"),
    ("Smoking",             "CardiovascularDisease"),
    ("Smoking",             "ChronicKidneyDisease"),
    ("PhysicalInactivity",  "Diabetes"),
    ("PhysicalInactivity",  "Obesity"),
]

# All unique node names used in the DAG
ALL_BN_NODES = list(dict.fromkeys(n for edge in SEED_EDGES for n in edge))

# Binary states for every node
STATES = ["absent", "present"]


class BayesianEngine:
    """
    Wraps a pgmpy DiscreteBayesianNetwork and exposes convenient methods for:
      - incremental CPT learning from patient records
      - risk inference given clinical evidence
      - serialisation to / deserialisation from Neo4j dict format
    """

    def __init__(self):
        self.model: DiscreteBayesianNetwork = DiscreteBayesianNetwork(SEED_EDGES)
        self._record_count: int = 0
        self._initialise_uniform_cpds()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise_uniform_cpds(self):
        """
        Seed every node with a uniform (50/50) CPT so the model is valid and
        queryable immediately — before any patient data has been ingested.
        """
        for node in self.model.nodes():
            parents = list(self.model.predecessors(node))
            n_parent_configs = 2 ** len(parents)
            # Uniform: P(present)=0.5 for every parent combination
            values = [[0.5] * n_parent_configs, [0.5] * n_parent_configs]
            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=values,
                evidence=parents if parents else None,
                evidence_card=[2] * len(parents) if parents else None,
                state_names={node: STATES, **{p: STATES for p in parents}},
            )
            self.model.add_cpds(cpd)

        if not self.model.check_model():
            logger.warning("BayesianEngine: seed CPD check failed — model may be inconsistent.")

    # ── Data binarisation ─────────────────────────────────────────────────────

    @staticmethod
    def _binarise_records(records: list[dict]) -> pd.DataFrame:
        """
        Convert raw Neo4j patient rows into a binary (absent/present) DataFrame
        matching the BN node names.

        Expected record keys: bmi, glucose, age, diagnosis_name (optional).
        """
        rows = []
        for r in records:
            bmi      = r.get("bmi") or 0
            glucose  = r.get("glucose") or 0
            age      = r.get("age") or 0
            diag     = (r.get("diagnosis_name") or "").lower()
            smoking  = (r.get("smoking") or "").lower()   # "yes"/"no" if available
            inactive = (r.get("inactive") or "").lower()  # "yes"/"no" if available

            rows.append({
                "Obesity":              "present" if bmi > BN_BMI_THRESHOLD      else "absent",
                "Diabetes":             "present" if ("diabet" in diag or glucose > BN_GLUCOSE_THRESHOLD) else "absent",
                "Hypertension":         "present" if "hypertens" in diag         else "absent",
                "Dyslipidemia":         "present" if "dyslipidem" in diag        else "absent",
                "ChronicKidneyDisease": "present" if ("ckd" in diag or "kidney" in diag) else "absent",
                "CardiovascularDisease":"present" if ("cardio" in diag or "cvd" in diag or "heart" in diag) else "absent",
                "Anemia":               "present" if "anemia" in diag            else "absent",
                "Smoking":              "present" if smoking == "yes"             else "absent",
                "PhysicalInactivity":   "present" if inactive == "yes"           else "absent",
            })

        return pd.DataFrame(rows)

    # ── Learning ──────────────────────────────────────────────────────────────

    def update_from_patient_data(self, records: list[dict]) -> int:
        """
        Re-estimate CPTs from all available patient records using Bayesian
        estimation with a Dirichlet prior (equivalent_sample_size = BN_DIRICHLET_PRIOR).

        pgmpy 1.0 requires:
          - DataFrame columns typed as pd.CategoricalDtype
          - model.fit(df, estimator=BayesianEstimator) rather than per-node estimate_cpd

        Returns the number of records used.
        """
        if not records:
            logger.info("BayesianEngine: no records provided — keeping current CPTs.")
            return 0

        df = self._binarise_records(records)
        self._record_count = len(df)

        try:
            # pgmpy 1.0: columns must be CategoricalDtype
            cat_dtype = pd.CategoricalDtype(categories=STATES, ordered=False)
            for col in df.columns:
                if col in self.model.nodes():
                    df[col] = df[col].astype(cat_dtype)

            # Fit using BayesianEstimator with BDeu prior
            self.model.fit(
                df,
                estimator=BayesianEstimator,
                prior_type="BDeu",
                equivalent_sample_size=BN_DIRICHLET_PRIOR,
                state_names={n: STATES for n in self.model.nodes()},
            )

            valid = self.model.check_model()
            logger.info(
                "BayesianEngine: CPTs updated from %d records. Model valid: %s",
                self._record_count, valid
            )
        except Exception as exc:
            logger.error("BayesianEngine: CPT update failed — %s", exc)

        return self._record_count


    # ── Inference ─────────────────────────────────────────────────────────────

    def query_risk(self, evidence: dict[str, str]) -> dict[str, dict[str, float]]:
        """
        Compute posterior risk for all non-evidence nodes given the supplied evidence.

        Args:
            evidence: e.g. {"Obesity": "present", "Smoking": "present"}
        Returns:
            dict of node → {"present": float, "absent": float}
        """
        # Only keep evidence variables that are actually in the model
        valid_evidence = {k: v for k, v in evidence.items() if k in self.model.nodes()}

        # pgmpy 1.0 with state_names: pass evidence as raw string states
        target_nodes = [n for n in self.model.nodes() if n not in valid_evidence]

        result: dict[str, dict[str, float]] = {}
        try:
            ve = VariableElimination(self.model)
            for node in target_nodes:
                phi = ve.query([node], evidence=valid_evidence, show_progress=False)
                # phi.state_names[node] gives the ordered states for this node
                ordered_states = phi.state_names[node]
                probs = phi.values.tolist()
                result[node] = {ordered_states[i]: round(probs[i], 4) for i in range(len(probs))}
        except Exception as exc:
            logger.error("BayesianEngine: inference failed — %s", exc)

        return result


    # ── Neo4j serialisation ───────────────────────────────────────────────────

    def to_neo4j_dict(self) -> dict[str, Any]:
        """
        Serialise the BN for storage in Neo4j.

        Returns a dict with:
          - "edges": list of {from, to}
          - "cpds": list of {node, parents, values_json}
          - "record_count": int
        """
        edges = [{"from": u, "to": v} for u, v in self.model.edges()]
        cpds = []
        for cpd in self.model.cpds:
            cpds.append({
                "node":        cpd.variable,
                "parents":     list(cpd.variables[1:]),
                "values_json": json.dumps(cpd.get_values().tolist()),
            })
        return {
            "edges":        edges,
            "cpds":         cpds,
            "record_count": self._record_count,
        }

    @classmethod
    def from_neo4j_dict(cls, data: dict[str, Any]) -> "BayesianEngine":
        """
        Reconstruct a BayesianEngine from data previously returned by to_neo4j_dict().
        """
        engine = cls.__new__(cls)
        edges = [(e["from"], e["to"]) for e in data.get("edges", [])]
        engine.model = DiscreteBayesianNetwork(edges or SEED_EDGES)
        engine._record_count = data.get("record_count", 0)

            # Re-attach CPDs
        for cpd_data in data.get("cpds", []):
            node    = cpd_data["node"]
            parents = cpd_data["parents"]
            values  = json.loads(cpd_data["values_json"])
            
            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=values,
                evidence=parents if parents else None,
                evidence_card=[2] * len(parents) if parents else None,
                state_names={node: STATES, **{p: STATES for p in parents}},
            )
            engine.model.add_cpds(cpd)
            engine.model.add_cpds(cpd)

        if not engine.model.check_model():
            logger.warning("BayesianEngine.from_neo4j_dict: model check failed after reload.")
            engine._initialise_uniform_cpds()

        return engine
