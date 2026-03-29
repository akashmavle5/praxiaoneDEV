@echo off
TITLE Praxia5Chronic - GraphRAG Deployment
echo ===================================================
echo 🚀 Launching Praxia5Chronic + Grafhir + Neo4j Environment
echo ===================================================

echo [1] Creating local directories for Neo4j (Grafhir structure)...
mkdir neo4j_data\data 2>nul
mkdir neo4j_data\logs 2>nul
mkdir neo4j_data\import 2>nul
mkdir neo4j_data\plugins 2>nul

echo [2] Starting Neo4j Vector-Enabled Container...
:: Note: Upgraded to neo4j:5.x from grafhir's 4.2 to support Vector Indexes for GraphRAG
docker run --name praxia5_neo4j -p 7474:7474 -p 7687:7687 -d ^
  -v %cd%\neo4j_data\data:/data ^
  -v %cd%\neo4j_data\logs:/logs ^
  -v %cd%\neo4j_data\import:/var/lib/neo4j/import ^
  -v %cd%\neo4j_data\plugins:/plugins ^
  --env NEO4J_AUTH=neo4j/praxia5password ^
  --env NEO4J_apoc_export_file_enabled=true ^
  --env NEO4J_apoc_import_file_enabled=true ^
  --env NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* ^
  neo4j:5.20.0

echo [3] Setting up Python Virtual Environment...
python -m venv venv
call venv\Scripts\activate

echo [4] Installing PJKG Dependencies...
pip install -r requirements.txt
echo.
echo [!] IMPORTANT: Make sure you have created your .env file with GEMINI_API_KEY!
echo [!] Make sure Ollama is running deepseek-r1:8b and Med42 before testing.
echo.

echo [5] Starting Praxia5Chronic FastAPI Server...
start uvicorn main:app --reload --port 8000

echo ===================================================
echo ✅ Setup Complete! 
echo 🌐 Neo4j Browser: http://localhost:7474 (neo4j / praxia5password)
echo ⚡ API Endpoint: http://localhost:8000/docs
echo ===================================================
pause