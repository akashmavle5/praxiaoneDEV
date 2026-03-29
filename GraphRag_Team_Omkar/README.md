# Praxia5Chronic - Multimodal Patient Journey Knowledge Graph (PJKG)

This project integrates various medical data sources (CSV, PDF, Docx, Text) into a Neo4j Graph Database using a custom Patient Journey Knowledge Graph ontology. It features a FastAPI backend, a powerful Neo4j Graph Database, and a GraphRAG interface for querying patient insights and dynamically updating a Bayesian Network for clinical risk prediction.

## 🚀 Key Features

*   **Multimodal Data Ingestion**:
    *   **Tabular Data**: Ingest structural CSV datasets (Diabetes, Heart Disease) mapping them automatically to Patient, Encounter, DiagnosticTest, and Diagnosis nodes (`ingest_diabetes_dataset.py`).
    *   **Unstructured Data**: Robust OCR and parsing of medical documents (PDFs, DOCX, images, TXT) via PyPDF, docx, pdfplumber, and Tesseract. Sentences are chunked and embedded via `sentence-transformers` into a local FAISS vector database (`pdf_ingestion.py`).
*   **LLM Medical Extractor Engine**:
    *   Uses local LLMs (Ollama with Med42/Mistral variants) to parse unstructured patient encounter transcripts.
    *   Extracts strictly validated JSON representing Diagnoses, Symptoms, Medications, Vitals, and Temporal patterns (`llm_extractor.py`, `validator.py`).
*   **Dynamic Bayesian Network (BN) for Risk Inference**:
    *   Continuously recalculates Conditional Probability Tables (CPTs) via pgmpy as new patients are ingested.
    *   Persists the BN DAG and CPTs directly into Neo4j.
    *   Provides realtime posterior probabilities for diseases (Obesity, Hypertension, Diabetes, etc.) given clinical evidence (BMI, Glucose, Age, etc.) (`bayesian_network.py`).
*   **GraphRAG & Smart Query Routing**:
    *   **Query Router**: Uses an LLM agent to classify user intents dynamically, routing them to the *Graph DB*, *Vector DB*, or a general query (`query_router.py`).
    *   **GraphRAG**: Parses natural language filters (e.g., "fasting glucose normal", "bmi obese") to build complex dynamic Cypher queries. Returns insights annotated with predictive Bayesian Risks (`graph_rag.py`).
    *   **Vector Querying**: Enables direct semantic search against ingested medical guidelines and patient discharge summaries via FAISS (`retriever.py`, `diagnosis_engine.py`).
*   **Parallel Multi-Model Pipeline (NEW)**:
    *   Queries submitted to the system simultaneously execute across three separate models: **Google Gemini 2.5**, **DeepSeek-R1 (Local)**, and **Med42 (Healthcare Local)**.
    *   Returns beautifully structured parallel UI cards to aggregate insights securely and privately.
*   **Clinical Rules & Standards**:
    *   Evaluate patient lab values instantly against standard US health ranges for BMI, Glucose, Blood Pressure, Cholesterol, and Sleep (`service.py`).

## 🛠 Project Setup & Installation

**🐳 Docker Deployment Available**: If you are deploying this application to a server, we highly recommend using the new Docker-Compose setup. Please read the [README-Docker.md](./README-Docker.md) for full instructions.

### 1. Prerequisites
- Python 3.9+
- A [Neo4j Aura](https://console.neo4j.io/) database instance (or local Neo4j desktop)
- Ollama installed locally (for LLM operations) with `deepseek-r1:8b` and `hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M` models pulled.
- A valid Google Gemini API Key.
- Tesseract OCR installed on your system (for scanned PDF/Image ingestion).

### 2. Installation
Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install requirements:
```bash
pip install fastapi uvicorn pydantic neo4j sentence-transformers python-multipart pandas langchain-community langchain-text-splitters langchain-huggingface langchain-ollama pgmpy pdfplumber pdf2image pytesseract python-docx faiss-cpu pypdf google-generativeai python-dotenv
```

Next, pull the required local LLM models using Ollama:
```bash
ollama run deepseek-r1:8b
ollama run hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M
```

### 3. Configuration
1. **Gemini API Setup**:
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your_google_gemini_api_key"
```

2. **Neo4j DB Setup**:
Update the `config.py` file with your Neo4j Aura database credentials and preferred settings:
```python
NEO4J_URI = "neo4j+s://<YOUR_DB_ID>.databases.neo4j.io"
NEO4J_USER = "<YOUR_AURA_USERNAME>"
NEO4J_PASSWORD = "<YOUR_AURA_PASSWORD>"

# You can also customise BN thresholds here
BN_GLUCOSE_THRESHOLD = 140
BN_BMI_THRESHOLD = 30
BN_AGE_THRESHOLD = 50
```

### 4. Running the Application
To start the FastAPI server and the frontend dashboard, run the following command in your terminal from the root directory:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Then, open your web browser and go to: **[http://localhost:8000](http://localhost:8000)**

## 💻 API Endpoints Reference

The FastAPI application (`main.py`) exposes several endpoints to interact with the system:

*   **`GET /`**: Serves the frontend web dashboard (`index.html`).
*   **`GET /api/health`**: Checks system status, DB connection, and Bayesian Network state.
*   **`POST /api/upload-dataset`**: Upload Tabular data (CSV) causing dynamic schema detection and ingestion, OR upload unstructured documents (PDF) to the Vector DB.
*   **`POST /api/upload-pdf` or `/upload-medical-file`**: Direct endpoints for medical document ingestion with built-in chunking, embedding, and FAISS indexing.
*   **`POST /api/ingest-encounter`**: Processes unstructured text snippets using Med42 LLM to populate the graph with Encounter and Diagnosis entities.
*   **`POST /chat`**: The primary conversational endpoint interacting with the Query Router. Auto-selects between Vector DB, Graph DB, or general medical knowledge.
*   **`POST /api/graph-rag`**: Direct interface for querying the Graph DB. Uses LLMs to generate appropriate Cypher statements and summarizes the data retrieved.
*   **`POST /api/bn-risk`**: Submit clinical evidence (e.g. `{"evidence": {"Obesity": "present", "Smoking": "present"}}`) to receive probabilistic posterior disease risks.
*   **`GET /api/bn-graph`**: Retrieve the topological graph structure of the Bayesian Network for frontend UI rendering.
*   **`POST /api/check-ranges`**: Programmatic endpoint to validate vitals (glucose, bmi, cholesterol, etc.) against embedded medical guidelines.

## 📊 Visualizing the Graph (Neo4j Console)

To see the visual graph representation, log into your Neo4j Aura console -> **Query** tab, and enter these Cypher queries:

**Basic Overview:**
```cypher
MATCH (n) RETURN n LIMIT 100
```

**Patient Journey View:**
```cypher
MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_DIAGNOSIS]->(d:Diagnosis) 
RETURN p, e, d 
LIMIT 50
```

**View Bayesian Network Structure:**
```cypher
MATCH (b:BayesianNode)-[r:BAYES_EDGE]->(c:BayesianNode)
RETURN b, r, c
```
