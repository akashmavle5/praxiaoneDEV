# LLM Gateway Layer (Mansa)

This module implements the LLM Layer – Unified Inference Abstraction.

## Features
- Ollama local inference
- Commercial model support
- Unified interface:
  llm.generate(prompt, reasoning_mode="graph", budget=0.02, safety="clinical")
- Model monitoring:
  - Drift
  - Variance
  - Failure tracking

## Files
build_llm_gateway.sh → Linux/Mac build script  
build_llm_gateway.bat → Windows build script  
gpt_output.txt → Run instructions  

## Run

### Linux/Mac
chmod +x build_llm_gateway.sh  
./build_llm_gateway.sh  

### Windows
build_llm_gateway.bat
