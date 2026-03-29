#!/bin/bash

echo "==================================================="
echo "🚀 Launching Praxia5Chronic + Grafhir + Neo4j Environment"
echo "==================================================="

echo "[1] Creating local directories for Neo4j (Grafhir structure)..."
mkdir -p neo4j_data/data neo4j_data/logs neo4j_data/import neo4j_data/plugins

echo "[2] Starting Neo4j Vector-Enabled Container..."
docker run --name praxia5_neo4j -p 7474:7474 -p 7687:7687 -d \
  -v $(pwd)/neo4j_data/data:/data \
  -v $(pwd)/neo4j_data/logs:/logs \
  -v $(pwd)/neo4j_data/import:/var/lib/neo4j/import \
  -v $(pwd)/neo4j_data/plugins:/plugins \
  --env NEO4J_AUTH=neo4j/praxia5password \
  --env NEO4J_apoc_export_file_enabled=true \
  --env NEO4J_apoc_import_file_enabled=true \
  --env NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* \
  neo4j:5.20.0

echo "[3] Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "[4] Installing PJKG Dependencies..."
pip install -r requirements.txt

echo ""
echo "[!] IMPORTANT: Ensure you have your Neo4j DB configured in config.py."
echo "[!] Make sure Ollama is running deepseek-r1:8b and Med42 before testing."
echo ""

echo "[5] Starting Praxia5Chronic FastAPI Server..."
uvicorn main:app --reload --port 8000
