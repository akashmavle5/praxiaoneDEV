#!/bin/bash

echo "Creating PROFESSIONAL ai_llm_gateway package..."

PACKAGE="ai_llm_gateway"

rm -rf $PACKAGE
rm -rf dist
rm -rf build
rm -rf *.egg-info

# =========================
# CREATE FOLDERS
# =========================
mkdir -p $PACKAGE/ai_llm_gateway/core
mkdir -p $PACKAGE/ai_llm_gateway/services
mkdir -p $PACKAGE/ai_llm_gateway/infrastructure/engines
mkdir -p $PACKAGE/ai_llm_gateway/infrastructure/monitoring
mkdir -p $PACKAGE/ai_llm_gateway/gateway

# =========================
# INIT
# =========================
cat > $PACKAGE/ai_llm_gateway/__init__.py << 'EOF'
from .gateway.llm_gateway import LLMGateway
EOF

# =========================
# CORE MODELS
# =========================
cat > $PACKAGE/ai_llm_gateway/core/models.py << 'EOF'
from pydantic import BaseModel

class GenerateResponse(BaseModel):
    text: str
    latency: float
    engine: str
EOF

# =========================
# MONITOR
# =========================
cat > $PACKAGE/ai_llm_gateway/infrastructure/monitoring/monitor_store.py << 'EOF'
import statistics

class MonitorStore:
    def __init__(self):
        self.sizes=[]
        self.failures=0

    def record(self,text:str):
        self.sizes.append(len(text))
        if "error" in text.lower():
            self.failures+=1

    def drift(self):
        if len(self.sizes)<2:return 0
        return max(self.sizes)-min(self.sizes)

    def variance(self):
        if len(self.sizes)<2:return 0
        return statistics.variance(self.sizes)

    def report(self):
        return {"drift":self.drift(),"variance":self.variance(),"failures":self.failures}
EOF

# =========================
# OLLAMA ENGINE
# =========================
cat > $PACKAGE/ai_llm_gateway/infrastructure/engines/ollama_engine.py << 'EOF'
import requests

class OllamaEngine:
    def __init__(self,model="llama3"):
        self.model=model
        self.url="http://localhost:11434/api/generate"

    def generate(self,prompt:str)->str:
        try:
            r=requests.post(self.url,json={"model":self.model,"prompt":prompt,"stream":False},timeout=60)
            return r.json().get("response","")
        except:
            return "ollama error"
EOF

# =========================
# COMMERCIAL ENGINE
# =========================
cat > $PACKAGE/ai_llm_gateway/infrastructure/engines/commercial_engine.py << 'EOF'
import requests

class CommercialEngine:
    def __init__(self,api_key):
        self.api_key=api_key

    def generate(self,prompt:str)->str:
        headers={"Authorization":f"Bearer {self.api_key}"}
        r=requests.post("https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}]})
        return r.json()["choices"][0]["message"]["content"]
EOF

# =========================
# SERVICE
# =========================
cat > $PACKAGE/ai_llm_gateway/services/llm_service.py << 'EOF'
import time
from ..core.models import GenerateResponse

class LLMService:
    def __init__(self,engine,monitor):
        self.engine=engine
        self.monitor=monitor

    def generate(self,prompt):
        start=time.time()
        text=self.engine.generate(prompt)
        latency=time.time()-start
        self.monitor.record(text)
        return GenerateResponse(text=text,latency=latency,engine=self.engine.__class__.__name__)
EOF

# =========================
# GATEWAY
# =========================
cat > $PACKAGE/ai_llm_gateway/gateway/llm_gateway.py << 'EOF'
from ..services.llm_service import LLMService
from ..infrastructure.monitoring.monitor_store import MonitorStore
from ..infrastructure.engines.ollama_engine import OllamaEngine
from ..infrastructure.engines.commercial_engine import CommercialEngine

class LLMGateway:

    def __init__(self,engine="ollama",api_key=None):
        if engine=="ollama":
            engine_instance=OllamaEngine()
        else:
            engine_instance=CommercialEngine(api_key)

        monitor=MonitorStore()
        self.service=LLMService(engine_instance,monitor)
        self.monitor=monitor

    def generate(self,prompt,reasoning_mode="graph",budget=0.02,safety="clinical"):
        res=self.service.generate(prompt)
        return {"response":res.text,
                "meta":{"engine":res.engine,"latency":res.latency,
                "mode":reasoning_mode,"budget":budget,"safety":safety}}

    def health(self):
        return self.monitor.report()
EOF

# =========================
# SETUP
# =========================
cat > $PACKAGE/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="ai_llm_gateway",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["requests","pydantic"],
)
EOF

cd $PACKAGE

echo "Building wheel..."
pip install wheel
python setup.py sdist bdist_wheel

echo "Installing..."
pip install dist/*.whl

echo "SUCCESS: Professional ai_llm_gateway installed"
