#!/bin/bash

set -e

ROOT="ai_cost_controller"
PKG="ai_cost_controller"
VERSION="1.0.0"

echo "🚀 Creating enterprise package structure..."

mkdir -p $ROOT
cd $ROOT

# -------------------------
# Create inner package folder
# -------------------------

mkdir -p $PKG/{execution,providers,governance,optimization,observability,persistence,utils}

touch $PKG/__init__.py

# -------------------------
# __init__.py
# -------------------------

cat > $PKG/__init__.py <<EOF
from .execution.orchestrator import Orchestrator
from .governance.policy_engine import PolicyEngine

__all__ = ["Orchestrator", "PolicyEngine"]
EOF

# -------------------------
# config.py
# -------------------------

cat > $PKG/config.py <<EOF
import os

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_cost.db")
EOF

# -------------------------
# EXECUTION
# -------------------------

cat > $PKG/execution/orchestrator.py <<EOF
class Orchestrator:
    def __init__(self, provider):
        self.provider = provider

    def run(self, model, prompt):
        return self.provider.call(model, prompt)
EOF

cat > $PKG/execution/circuit_breaker.py <<EOF
import time

class CircuitBreaker:
    def __init__(self, threshold=3, timeout=30):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.threshold:
            self.state = "OPEN"

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def allow(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True
EOF

# -------------------------
# PROVIDERS
# -------------------------

cat > $PKG/providers/base.py <<EOF
class BaseProvider:
    def call(self, model, prompt):
        raise NotImplementedError
EOF

cat > $PKG/providers/groq.py <<EOF
from groq import Groq
import time
from ..config import Config

class GroqProvider:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)

    def call(self, model, prompt):
        start = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = tokens / 1000 * 0.0005
        return text, tokens, cost, latency
EOF

cat > $PKG/providers/ollama.py <<EOF
import requests
import time

class OllamaProvider:
    BASE_URL = "http://localhost:11434"

    def call(self, model, prompt):
        start = time.time()
        r = requests.post(
            f"{self.BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        latency = time.time() - start
        data = r.json()
        text = data.get("response", "")
        tokens = len(text.split())
        cost = 0.0
        return text, tokens, cost, latency
EOF

# -------------------------
# GOVERNANCE
# -------------------------

cat > $PKG/governance/policy_engine.py <<EOF
class PolicyEngine:
    def __init__(self):
        self.usage = {}
        self.limits = {}

    def set_limit(self, user, tokens):
        self.limits[user] = tokens

    def allow(self, user, tokens):
        used = self.usage.get(user, 0)
        if user in self.limits and used + tokens > self.limits[user]:
            return False
        return True

    def record(self, user, tokens):
        self.usage[user] = self.usage.get(user, 0) + tokens
EOF

# -------------------------
# OPTIMIZATION
# -------------------------

cat > $PKG/optimization/roi.py <<EOF
class ROIEngine:
    def compute(self, cost, value):
        if cost == 0:
            return value
        return value / cost
EOF

# -------------------------
# OBSERVABILITY
# -------------------------

cat > $PKG/observability/logging.py <<EOF
import logging

def get_logger():
    logger = logging.getLogger("ai_cost")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger
EOF

# -------------------------
# PERSISTENCE
# -------------------------

cat > $PKG/persistence/models.py <<EOF
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class LLMLog(Base):
    __tablename__ = "llm_logs"
    id = Column(Integer, primary_key=True)
    provider = Column(String)
    tokens = Column(Integer)
    cost = Column(Float)