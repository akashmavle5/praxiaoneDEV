#!/bin/bash

echo "🚀 Building and starting Praxia5Chronic Docker containers..."
docker-compose up -d --build

echo "⏳ Waiting 10 seconds for Ollama service to fully initialize..."
sleep 10

echo "🧠 Downloading DeepSeek-R1 (8B)..."
docker exec -it updated_internship_gemini-ollama-1 ollama run deepseek-r1:8b

echo "🩺 Downloading Med42 Healthcare Model..."
docker exec -it updated_internship_gemini-ollama-1 ollama run hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M

echo "✅ Deployment complete! The GraphRAG Dashboard is available at http://localhost:8000"
