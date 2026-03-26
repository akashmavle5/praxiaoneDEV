from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import os
from service import PJKGService
from graph_rag import PraxiaGraphRAG
from bayesian_network import BayesianEngine
from ingest_diabetes_dataset import DiabetesIngester, ingest_heart_dataset
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import logging
import shutil
import json
import time
import re
import pandas as pd
from openai import OpenAI
from query_router import classify_query
from retriever import ask_llm
from pdf_ingestion import extract_text
from pdf_ingestion import ingest_pdf
from diagnosis_engine import diagnose
from retriever import db
from pdf_ingestion import search_pdf
from langchain_community.llms import Ollama
llm = Ollama(model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M")

import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
try:
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print(f"Gemini initialization error: {e}")
    gemini_model = None

deepseek_llm = Ollama(model="deepseek-r1:8b")

async def ask_deepseek_async(question, context, is_general=False):
    if is_general:
        prompt = f"""You are a highly knowledgeable medical and wellness AI assistant.
The user is asking a general health or educational inquiry: "{question}"
CRITICAL INSTRUCTIONS: Answer directly and comprehensively. Use clear markdown formatting."""
    else:
        prompt = f"""Use the following document text to answer the question:
Context: {context}
Question: {question}
Instructions: Answer concisely based ONLY on the context. Format using professional Markdown tables if requested."""
    try:
        res = await asyncio.to_thread(deepseek_llm.invoke, prompt)
        return str(res)
    except Exception as e:
        return f"DeepSeek Error: {str(e)}"

async def ask_gemini_async(question, context, is_general=False):
    if not gemini_model:
        return "Gemini API is not configured properly."
    if is_general:
        prompt = f"""You are a highly knowledgeable medical and wellness AI assistant.
The user is asking a general health or educational inquiry: "{question}"
CRITICAL INSTRUCTIONS: Answer directly and comprehensively. Use clear markdown formatting."""
    else:
        prompt = f"""Use the following document text to answer the question:
Context: {context}
Question: {question}
Instructions: Answer concisely based ONLY on the context. Format using professional Markdown tables if requested."""
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

async def ask_med42_async(question, context, is_general=False):
    if is_general:
        prompt = f"""You are a highly knowledgeable medical and wellness AI assistant.
The user is asking a general health or educational inquiry: "{question}"
CRITICAL INSTRUCTIONS: Answer directly and comprehensively. Use clear markdown formatting."""
    else:
        prompt = f"""Use the following document text to answer the question:
Context: {context}
Question: {question}
Instructions: Answer concisely based ONLY on the context. Format using professional Markdown tables if requested."""
    try:
        res = await asyncio.to_thread(llm.invoke, prompt)
        return str(res)
    except Exception as e:
        return f"Med42 Error: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.shutdown_tracker = {"is_shutting_down": False}
    yield
    # Shutdown definition 
    print("Application shutdown initiated. Signalling background tasks to stop...")
    app.state.shutdown_tracker["is_shutting_down"] = True

# -----------------------------
# ADDED: Glucose scale detection
# -----------------------------
def detect_glucose_scale(graph):
    """
    Detect whether glucose values are normalized (0–1) or mg/dL.
    """
    try:
        with graph.driver.session() as session:
            result = session.run("""
            MATCH (t:DiagnosticTest)
            WHERE t.glucose IS NOT NULL
            RETURN max(t.glucose) AS max_glucose
            LIMIT 1
            """)
            record = result.single()

            if record and record["max_glucose"] is not None:
                if record["max_glucose"] <= 1:
                    return "normalized"
                else:
                    return "mgdl"
    except Exception:
        pass

    return "unknown"
    

service = PJKGService()
# Suppress noisy expected transient connection warnings from Neo4j pool
logging.getLogger("neo4j").setLevel(logging.CRITICAL)

app = FastAPI(title="Praxia5Chronic Production API 🚀", lifespan=lifespan)

# 1. Initialize the services (Wrapped in try-except to prevent crash on invalid DB credentials)
pjkg_service = None
graph_rag = None

try:
    pjkg_service = PJKGService()
    bn_engine = pjkg_service.bn_engine
    graph_rag = PraxiaGraphRAG(pjkg_service.graph, bn_engine=bn_engine)
    # Hot-load the BN from the service (already loaded inside PJKGService.__init__)
except Exception as e:
    print(f"CRITICAL: Failed to initialize database connection. {e}")
    bn_engine = BayesianEngine()  # Fallback: seed-only BN with no data

# --- API Data Models ---
class EncounterData(BaseModel):
    patient_id: str
    encounter_id: str
    transcript: str
    date: str

class QueryData(BaseModel):
    query: str

class DatasetRequest(BaseModel):
    dataset: str

class ChatRequest(BaseModel):
    question: str
    model_choice: str = "ollama"
    active_file: str = None
class BNQueryData(BaseModel):
    """Evidence variables for Bayesian risk query.
    Keys must match BN node names (e.g. 'Obesity', 'Smoking').
    Values must be 'present' or 'absent'.
    """
    evidence: dict[str, str]


def format_text(text: str):

    if not text:
        return ""

    text = text.replace(". ", ".\n\n")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    return text
# --- API Endpoints ---
@app.get("/api/health")
def health_check():
    db_status = "Connected" if pjkg_service and pjkg_service.graph else "DISCONNECTED (Check credentials)"
    bn_records = bn_engine._record_count if bn_engine else 0
    return {
        "status": "Praxia5Chronic Cloud Engine is running! 🚀",
        "database": f"Neo4j Aura ({db_status})",
        "bayesian_network": f"Active — trained on {bn_records} patient records",
    }

@app.post("/api/bn-risk")
def query_bn_risk(data: BNQueryData):
    """
    Probabilistic risk inference using the Bayesian Network.
    Supply an evidence dict with known clinical states; the endpoint returns
    posterior probabilities for all remaining disease nodes.

    Example body:
      { "evidence": { "Obesity": "present", "Smoking": "present" } }
    """
    if bn_engine is None:
        return {"error": "Bayesian Network is not initialised."}

    valid_states = {"present", "absent"}
    bad = {k: v for k, v in data.evidence.items() if v not in valid_states}
    if bad:
        return {
            "error": f"Invalid state values for: {bad}. Use 'present' or 'absent'."
        }

    risk_scores = bn_engine.query_risk(data.evidence)
    record_count = bn_engine._record_count

    return {
        "evidence":     data.evidence,
        "risk_scores":  risk_scores,
        "explanation":  (
            f"Probabilities computed by Variable Elimination over the clinical "
            f"Bayesian Network (trained on {record_count} patient records)."
        ),
    }

@app.get("/api/bn-graph")
def get_bn_graph():
    """Returns the nodes and edges of the Bayesian Network for frontend visualization."""
    if bn_engine is None:
        return {"error": "Bayesian Network is not initialised."}

    try:
        from pgmpy.inference import VariableElimination
        ve = VariableElimination(bn_engine.model)
        
        nodes = []
        for node in bn_engine.model.nodes():
            try:
                # Get marginal probability with no evidence
                phi = ve.query([node], show_progress=False)
                states = phi.state_names[node]
                values = phi.values.tolist()
                
                prob_present = 0.0
                if "present" in states:
                    idx = states.index("present")
                    prob_present = values[idx]
                
                nodes.append({
                    "id": node,
                    "label": f"{node}\n{(prob_present * 100):.1f}% risk",
                    "prob": prob_present,
                    "title": f"Probability of being present: {(prob_present * 100):.1f}%"
                })
            except Exception as e:
                # Fallback if inference fails
                nodes.append({
                    "id": node,
                    "label": node,
                    "prob": 0.5,
                    "title": "Marginal probability unavailable"
                })
                
        edges = [{"from": u, "to": v} for u, v in bn_engine.model.edges()]
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/upload-dataset")
async def upload_dataset(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        content = await file.read()
        temp_path = os.path.join(os.path.dirname(__file__), file.filename)

        with open(temp_path, "wb") as f:
            f.write(content)

        # -------- PDF Handling --------
        if file.filename.lower().endswith(".pdf"):
            background_tasks.add_task(ingest_pdf, temp_path)
            return {"message": "PDF uploaded and ingestion started."}

        # -------- CSV Handling --------
        import pandas as pd
        df = pd.read_csv(temp_path, nrows=5)
        
        # Pass the mutable tracker dict to the tasks
        stop_state = request.app.state.shutdown_tracker

        # Schema Detection
        if "Glucose" in df.columns or "BMI" in df.columns:
            ingester = DiabetesIngester(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            background_tasks.add_task(ingester.run_ingestion, temp_path, stop_state)
            return {"message": "Detected Diabetes schema. Ingestion started in the background."}

        elif "Age" in df.columns and "RestingBP" in df.columns and "HeartDisease" in df.columns:
            background_tasks.add_task(ingest_heart_dataset, temp_path, stop_state)
            return {"message": "Detected Heart Disease schema. Ingestion started in the background."}

        else:
            from ingest_diabetes_dataset import generic_ingest_dataset
            background_tasks.add_task(generic_ingest_dataset, temp_path, stop_state)
            columns = list(df.columns)
            return {"message": f"Detected Generic Dataset ({len(columns)} columns). Generic inference ingestion started in the background."}

    except Exception as e:
        return {"error": str(e)}

@app.post("/api/ingest-encounter")
def ingest_encounter(data: EncounterData):
    """Processes a raw medical transcript, extracts entities via Med42, and builds the Graph."""
    if pjkg_service is None:
        return {"status": "error", "message": "Database service is not initialized. Please check Neo4j credentials in config.py."}
    
    result = pjkg_service.process_patient_encounter(
        patient_id=data.patient_id,
        encounter_id=data.encounter_id,
        transcript=data.transcript,
        date=data.date
    )
    return result

@app.post("/api/graph-rag")
def query_graph_rag(data: QueryData):
    """Dynamic Conversational Agent identifying cypher questions vs Bayesian risk."""
    if pjkg_service is None or pjkg_service.graph is None:
        return {"error": "GraphRAG engine is not initialized. Please check Neo4j credentials in config.py."}

    if not data.query:
        return {"error": "No query provided"}

    try:
        from openai import OpenAI
        import json
        llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # ADDED: detect glucose scale dynamically
        glucose_scale = detect_glucose_scale(pjkg_service.graph)
        # 1. Classify Intent
        classification_prompt = f"""
You are a medical AI assistant. Classify the user query into ONE of three intents:
    Dataset glucose scale: {glucose_scale}

If glucose values are normalized (0-1), treat high glucose as > 0.7
If glucose values are mg/dL, treat high glucose as > 100
1. 'cypher': The user asks for database statistics, counts, or explicit patient records (e.g., "how many people have diabetes?", "give me 10 people with heart disease", "how many people are in the database").
   - Output 'cypher' property containing a valid Neo4j Cypher query using this schema:
     (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_TEST]->(t:DiagnosticTest)
     (e)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
     (e)-[:HAS_VITALS]->(v:VitalSign)
     Node Properties: Patient(age, bmi, sex), DiagnosticTest(glucose, insulin), Diagnosis(name), VitalSign(blood_pressure, bmi)
   - CRITICAL: The query MUST start with the MATCH keyword. (e.g., "MATCH (p:Patient) RETURN count(p)")
   - CRITICAL: Do NOT declare python types like `float` or `string` in your Cypher query. Use raw property names. For example, use `WHERE v.bmi < 100` instead of `{{bmi: float}}`.
   - (no limit in query). Make it read-only. Ensure `toLower(d.name) CONTAINS` when filtering conditions.
2. 'bayesian': The user asks about risk based on their symptoms (e.g., "what are my chances of diabetes if my bmi is below 100").
   - Output 'evidence' property with extracted symptoms. Valid keys: 'Obesity', 'Diabetes', 'Hypertension', 'Dyslipidemia', 'ChronicKidneyDisease', 'CardiovascularDisease', 'Anemia', 'Smoking', 'PhysicalInactivity'.
   - Values: 'present', 'absent'. e.g., BMI < 30 -> Obesity: absent. BMI > 30 -> Obesity: present.
3. 'general': The user asks for general medical advice or information (e.g., "how can people take care of their health?", "how do I lower my blood pressure?").
   - No additional properties needed.
       - CRITICAL: Never return entire nodes like RETURN p.
   - Always return readable properties instead.
   - IMPORTANT CYPHER RULES:

     1. NEVER place conditions inside relationships.

     WRONG:
     MATCH (p)-[:HAS_TEST {{glucose > 1}}]->(t)

     CORRECT:
     MATCH (p)-[:HAS_TEST]->(t)
     WHERE t.glucose > 1

     2. Always use WHERE clause for filtering conditions.

     3. Correct relationship names are:
        HAS_ENCOUNTER
        HAS_TEST
        HAS_DIAGNOSIS
        HAS_VITALS

     4. Always return readable properties instead of nodes.

     WRONG:
     RETURN p

     CORRECT:
     RETURN p.patient_id AS patient_id
     
     5. IMPORTANT: Do NOT chain multiple encounters together backwards (e.g. `<-(:Encounter)<-(e:Encounter)` is INVALID).
     Use simple forward paths. To match diagnosis:
     MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
     WHERE toLower(d.name) CONTAINS 'diabetes'
    Example:
MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_TEST]->(t:DiagnosticTest)
WHERE t.glucose > 0.7
RETURN p.patient_id AS patient_id, p.age AS age, p.bmi AS bmi, t.glucose AS glucose
LIMIT 50
Respond strictly with valid JSON.
Format example:
{{"intent": "cypher", "cypher": "MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_TEST]->(t:DiagnosticTest) RETURN p.patient_id AS patient_id, p.age AS age, p.bmi AS bmi, t.glucose AS glucose LIMIT 50"}}
User query: "{data.query}"
"""
        response = llm_client.chat.completions.create(
            model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.0,
            max_tokens=300
        )
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        
        intent = "general"
        parsed = {}
        if start != -1 and end != 0:
            json_str = content[start:end].replace("`", "").strip()
            try:
                parsed = json.loads(json_str)
                intent = parsed.get("intent", "general")
            except Exception:
                pass
        
        if intent == "cypher" and "cypher" in parsed:
            cypher_query = parsed["cypher"]
            try:
                import time
                records = []
                for attempt in range(3):
                    try:
                        with pjkg_service.graph.driver.session() as session:
                            res = session.run(cypher_query)
                            # ADDED: large dataset safeguard
                            records = []
                            for i, r in enumerate(res):
                                records.append(dict(r))
                                if i > 5000:
                                    break
                            break
                    except Exception as neo_err:
                        print(f"Neo4j query attempt {attempt+1}/3 failed: {neo_err}")
                        time.sleep(1)
                        if attempt == 2:
                            raise neo_err
                    
                ans_prompt = f"""
You are a clinical data assistant.

A user asked the following question:
{data.query}

The medical database returned the following records from Neo4j.

IMPORTANT INSTRUCTIONS:
-Provide human-friendly answer.
- Use ONLY the information from the database records below.
- Do NOT invent or assume any information.
- If patient records exist, explain the result clearly and summarize them.
- Mention how many records were found.
- If no records exist, clearly say that no matching patients were found.
- Write the answer in simple, human-friendly language.

Database result:
Total records found: {len(records)}

Sample records:
{records[:10]}

Provide a clear explanation of the result for the user.
"""
                ans_response = llm_client.chat.completions.create(
                    model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
                    messages=[{"role": "user", "content": ans_prompt}],
                    temperature=0.0,
                    max_tokens=500
                )
                return {
                    "query": data.query,
                    "message": ans_response.choices[0].message.content,
                    "retrieved_graph_context": f"Cypher matched: {cypher_query}\nResult shape: {len(records)} objects"
                }
            except Exception as e:
                return {
                    "query": data.query,
                    "message": f"Execution of internal query failed: {e}",
                    "retrieved_graph_context": cypher_query
                }
                
        elif intent == "bayesian" and "evidence" in parsed:
            evidence = parsed["evidence"]
            if bn_engine:
                try:
                    # Filter out unrecognised nodes hallucinated by LLM (like 'bmi')
                    valid_nodes = set(bn_engine.model.nodes())
                    valid_evidence = {k: v for k, v in evidence.items() if k in valid_nodes and v in ["present", "absent"]}
                    
                    if not valid_evidence:
                         return {"query": data.query, "message": f"Could not map your symptoms to the clinical Bayesian Network (received {evidence}). Try asking about Obesity, Diabetes, or Hypertension.", "retrieved_graph_context": ""}

                    risk_scores = bn_engine.query_risk(valid_evidence)
                    risk_txt = "\\n".join([f"- {n}: {scores.get('present', 0)*100:.1f}%" for n, scores in risk_scores.items()])
                    
                    ans_prompt = f"User asked: {data.query}\nExtracted Evidence: {evidence}\nBayesian Network Risk Scores:\n{risk_txt}\nProvide a highly concise, colloquial clinical explanation of their risk. Do not explain probability math."
                    ans_response = llm_client.chat.completions.create(
                        model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
                        messages=[{"role": "user", "content": ans_prompt}],
                        temperature=0.7,
                        max_tokens=300
                    )
                    return {
                        "query": data.query,
                        "message": ans_response.choices[0].message.content,
                        "retrieved_graph_context": f"Evidence evaluated: {evidence}\nComputed Risks:\n{risk_txt}"
                    }
                except Exception as e:
                    return {"query": data.query, "message": f"Bayesian risk eval failed: {e}", "retrieved_graph_context": ""}
            else:
                return {"query": data.query, "message": "Bayesian engine not ready.", "retrieved_graph_context": ""}
                
        else:
            context = graph_rag.retrieve_context(data.query) if graph_rag else ""
            rag_prompt = f"You are a clinical data analyst. Use the below data to answer: {data.query}\nContext: {context}"
            ans_response = llm_client.chat.completions.create(
                model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.0,
                max_tokens=500
            )
            return {
                "query": data.query,
                "message": ans_response.choices[0].message.content,
                "retrieved_graph_context": context
            }
            
    except Exception as e:
        return {"query": data.query, "message": f"LLM logic error: {e}", "retrieved_graph_context": ""}


@app.post("/api/check-ranges")
async def check_ranges(payload: dict):
    result = service.check_clinical_ranges(payload)
    return {
        "status": "success",
        "ranges": result
    }

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)

    # save file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # extract text
    from pdf_ingestion import extract_text
    text = extract_text(file_path)

    # store in vector DB
    from retriever import db
    db.add_texts([text])

    # SAVE vector database
    db.save_local("vector_store")

    return {
        "status": "indexed",
        "filename": file.filename,
        "saved_to": file_path
    }
@app.post("/api/diagnose")
async def diagnose_patient(query: str):
    result = diagnose(query)
    return {"diagnosis": result}

# Maintain an in-memory dictionary. For production, ideally use Redis or DB.
DOCUMENT_CACHE = {}

@app.post("/ask_pdf")
async def ask_pdf(question: str):
    answer = ask_llm(question)
    return {"answer": answer}  

@app.post("/upload-medical-file")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = extract_text(path)

    # Add to in-memory cache
    DOCUMENT_CACHE[file.filename] = text

    return {
        "status": "indexed",
        "file": file.filename
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):

    question = request.question
    model_choice = getattr(request, "model_choice", "ollama")
    active_file = getattr(request, "active_file", None)

    intent = classify_query(question)

    print("ROUTED TO:", intent)

    if intent == "FILE":
        
        context = ""
        if active_file and active_file in DOCUMENT_CACHE:
            context = DOCUMENT_CACHE[active_file]
        else:
            context = search_pdf(question)
        
        if not context or "No documents indexed" in context or str(context).startswith("Document search error"):
            return {
                "answer": "No document selected or found.",
                "is_parallel": False
            }
        else:
            deepseek_task = ask_deepseek_async(question, context, is_general=False)
            gemini_task = ask_gemini_async(question, context, is_general=False)
            med42_task = ask_med42_async(question, context, is_general=False)

            ds_res, gem_res, med42_res = await asyncio.gather(deepseek_task, gemini_task, med42_task, return_exceptions=True)

            return {
                "answer": "Parallel response complete.",
                "is_parallel": True,
                "deepseek": str(ds_res),
                "gemini": str(gem_res),
                "med42": str(med42_res)
            }

    elif intent == "GRAPH":
        # Graph query remains synchronous in a thread to wait for neo4j, but we can call it directly
        result = query_graph_rag(QueryData(query=question))

        if "message" in result:
            answer = result["message"]
        else:
            answer = "Graph query failed."

        answer = f"**🧠 GRAPH ANSWER:**\n\n{answer}"
        return {"answer": answer, "is_parallel": False}

    else:
        deepseek_task = ask_deepseek_async(question, "", is_general=True)
        gemini_task = ask_gemini_async(question, "", is_general=True)
        med42_task = ask_med42_async(question, "", is_general=True)

        ds_res, gem_res, med42_res = await asyncio.gather(deepseek_task, gemini_task, med42_task, return_exceptions=True)

        return {
            "answer": "Parallel response complete.",
            "is_parallel": True,
            "deepseek": str(ds_res),
            "gemini": str(gem_res),
            "med42": str(med42_res)
        }