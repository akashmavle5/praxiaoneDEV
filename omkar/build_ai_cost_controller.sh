#!/bin/bash

echo "Creating ai_cost_controller package..."

PACKAGE="ai_cost_controller"

# Clean old

rm -rf $PACKAGE
rm -rf dist
rm -rf build
rm -rf *.egg-info

# Create structure

mkdir -p $PACKAGE/$PACKAGE/providers

# ---------------- ROOT FILES ----------------

cat > $PACKAGE/requirements.txt <<EOF
openai>=1.0.0
anthropic>=0.25.0
google-generativeai>=0.5.0
requests
pydantic
sqlalchemy
tiktoken
wheel
setuptools
EOF

cat > $PACKAGE/setup.py <<EOF
from setuptools import setup, find_packages

setup(
name="ai_cost_controller",
version="1.0.0",
description="Enterprise AI cost controller + model router",
author="AI Enterprise",
packages=find_packages(),
install_requires=[
"openai",
"anthropic",
"google-generativeai",
"requests",
"pydantic",
"sqlalchemy",
"tiktoken"
],
python_requires=">=3.9",
)
EOF

cat > $PACKAGE/pyproject.toml <<EOF
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
EOF

cat > $PACKAGE/README.md <<EOF
AI Cost Controller
Enterprise LLM cost governance + model router.
EOF

# ---------------- INIT ----------------

cat > $PACKAGE/$PACKAGE/**init**.py <<EOF
from .cost_controller import AICostController
EOF

# ---------------- CONFIG ----------------

cat > $PACKAGE/$PACKAGE/config.py <<EOF
import os

class Config:
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROK_KEY = os.getenv("GROK_API_KEY")
DB_URL = os.getenv("AI_COST_DB", "sqlite:///ai_cost.db")
EOF

# ---------------- MODELS ----------------

cat > $PACKAGE/$PACKAGE/models.py <<EOF
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import Config

Base = declarative_base()
engine = create_engine(Config.DB_URL)
Session = sessionmaker(bind=engine)

class LLMCallLog(Base):
**tablename** = "llm_logs"

```
id = Column(Integer, primary_key=True)
provider = Column(String)
model = Column(String)
tokens = Column(Integer)
cost = Column(Float)
latency = Column(Float)
accuracy_delta = Column(Float)
business_score = Column(Float)
```

def init_db():
Base.metadata.create_all(engine)
EOF

# ---------------- POLICY ENGINE ----------------

cat > $PACKAGE/$PACKAGE/policy_engine.py <<EOF
class TokenPolicyEngine:

```
def __init__(self):
    self.daily_cap = {}
    self.user_usage = {}
    self.project_usage = {}
    self.task_budget = {}

def set_daily_cap(self, user, tokens):
    self.daily_cap[user] = tokens

def set_task_budget(self, task, tokens):
    self.task_budget[task] = tokens

def check_allowed(self, user, project, task, tokens_needed):
    if user in self.daily_cap:
        if self.user_usage.get(user, 0) + tokens_needed > self.daily_cap[user]:
            return False, "Daily cap exceeded"

    if task in self.task_budget:
        if tokens_needed > self.task_budget[task]:
            return False, "Task budget exceeded"

    return True, "Allowed"

def record_usage(self, user, project, tokens):
    self.user_usage[user] = self.user_usage.get(user, 0) + tokens
    self.project_usage[project] = self.project_usage.get(project, 0) + tokens
```

EOF

# ---------------- ROI ENGINE ----------------

cat > $PACKAGE/$PACKAGE/roi_engine.py <<EOF
from .models import LLMCallLog, Session

class ROIEngine:

```
def log_call(self, provider, model, tokens, cost, latency, accuracy_delta, business_score):
    session = Session()
    log = LLMCallLog(
        provider=provider,
        model=model,
        tokens=tokens,
        cost=cost,
        latency=latency,
        accuracy_delta=accuracy_delta,
        business_score=business_score
    )
    session.add(log)
    session.commit()
    session.close()

def compute_roi(self, cost, business_score):
    if cost == 0:
        return business_score
    return business_score / cost
```

EOF

# ---------------- MODEL SWITCHER ----------------

cat > $PACKAGE/$PACKAGE/model_switcher.py <<EOF
class ModelSwitcher:

```
def __init__(self):
    self.task_map = {
        "draft": ("ollama", "llama3"),
        "verify": ("openai", "gpt-4o-mini"),
        "final": ("openai", "gpt-4o")
    }

def get_model(self, task_type, budget_priority="balanced"):
    if budget_priority == "cheap":
        return ("ollama", "llama3")
    if budget_priority == "premium":
        return ("openai", "gpt-4o")
    return self.task_map.get(task_type, ("ollama", "llama3"))
```

EOF

# ---------------- OLLAMA PROVIDER ----------------

cat > $PACKAGE/$PACKAGE/providers/ollama_provider.py <<EOF
import requests, time, json

class OllamaProvider:

```
def __init__(self, base_url="http://localhost:11434"):
    self.base_url = base_url.rstrip("/")

def _estimate_tokens(self, text: str) -> int:
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def list_models(self):
    url = f"{self.base_url}/api/tags"
    r = requests.get(url)
    data = r.json()
    return [m["name"] for m in data.get("models", [])]

def call(self, model: str, prompt: str, stream: bool = False):
    url = f"{self.base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    start = time.time()

    r = requests.post(url, json=payload)
    latency = time.time() - start

    if r.status_code != 200:
        raise Exception(r.text)

    data = r.json()
    text = data.get("response", "")
    tokens = self._estimate_tokens(prompt) + self._estimate_tokens(text)
    cost = 0.0

    return text, tokens, cost, latency
```

EOF

# ---------------- COST CONTROLLER ----------------

cat > $PACKAGE/$PACKAGE/cost_controller.py <<EOF
from .policy_engine import TokenPolicyEngine
from .roi_engine import ROIEngine
from .model_switcher import ModelSwitcher
from .providers.ollama_provider import OllamaProvider

class AICostController:

```
def __init__(self):
    self.policy = TokenPolicyEngine()
    self.roi = ROIEngine()
    self.switcher = ModelSwitcher()

    self.providers = {
        "ollama": OllamaProvider()
    }

def run_task(self, user, project, task_type, prompt, business_value=1.0):

    provider_name, model = self.switcher.get_model(task_type)
    provider = self.providers[provider_name]

    allowed, reason = self.policy.check_allowed(user, project, task_type, 5000)
    if not allowed:
        raise Exception(reason)

    text, tokens, cost, latency = provider.call(model, prompt)
    self.policy.record_usage(user, project, tokens)

    roi = self.roi.compute_roi(cost, business_value)

    self.roi.log_call(
        provider_name,
        model,
        tokens,
        cost,
        latency,
        accuracy_delta=roi,
        business_score=business_value
    )

    return {
        "output": text,
        "tokens": tokens,
        "cost": cost,
        "latency": latency,
        "roi": roi,
        "provider": provider_name,
        "model": model
    }
```

EOF

echo "Building wheel..."

cd $PACKAGE
pip install --upgrade build wheel setuptools >/dev/null 2>&1
python3 -m build >/dev/null 2>&1

echo ""
echo "BUILD COMPLETE"
echo "Wheel located in:"
echo "$PACKAGE/dist/"
echo ""
echo "Install using:"
echo "pip install dist/*.whl"
