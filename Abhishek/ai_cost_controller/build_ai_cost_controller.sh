#!/bin/bash

set -e

ROOT="ai_cost_controller"
PKG="ai_cost_controller"
VERSION="2.0.0"

echo "🚀 Creating FULL Enterprise AI Cost Controller..."

mkdir -p $ROOT
cd $ROOT

# =========================================================
# Create Deep Architecture
# =========================================================

mkdir -p $PKG/{execution,providers,governance,optimization,observability,persistence,billing,cache,utils}

touch $PKG/__init__.py

# =========================================================
# __init__.py
# =========================================================

cat > $PKG/__init__.py <<EOF
from .execution.orchestrator import Orchestrator
from .governance.policy_engine import PolicyEngine
from .optimization.roi_engine import ROIEngine

__all__ = ["Orchestrator", "PolicyEngine", "ROIEngine"]
EOF

# =========================================================
# config.py
# =========================================================

cat > $PKG/config.py <<EOF
import os

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_cost.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
EOF

# =========================================================
# EXECUTION LAYER
# =========================================================

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

cat > $PKG/execution/retry.py <<EOF
import time
import random

class RetryExecutor:
    def __init__(self, retries=3, base_delay=0.5):
        self.retries = retries
        self.base_delay = base_delay

    def execute(self, fn):
        for attempt in range(self.retries):
            try:
                return fn()
            except Exception as e:
                if attempt == self.retries - 1:
                    raise e
                delay = self.base_delay * (2 ** attempt)
                jitter = random.uniform(0, 0.1)
                time.sleep(delay + jitter)
EOF

cat > $PKG/execution/orchestrator.py <<EOF
class Orchestrator:
    def __init__(self, router, fallback, quota, billing, roi, logger):
        self.router = router
        self.fallback = fallback
        self.quota = quota
        self.billing = billing
        self.roi = roi
        self.logger = logger

    async def execute(self, context, prompt):
        provider_chain = self.router.resolve(context)
        text, tokens, cost, latency = await self.fallback.execute_async(
            provider_chain,
            prompt
        )

        self.quota.validate(context.tenant_id, tokens, cost)
        self.billing.meter(context.tenant_id, tokens)
        roi_score = self.roi.compute(cost, context.business_value)

        self.logger.info(
            f"Tenant={context.tenant_id} Tokens={tokens} Cost={cost}"
        )

        return {
            "output": text,
            "tokens": tokens,
            "cost": cost,
            "latency": latency,
            "roi": roi_score
        }
EOF

# =========================================================
# PROVIDERS
# =========================================================

cat > $PKG/providers/base.py <<EOF
from ..execution.circuit_breaker import CircuitBreaker

class BaseProvider:
    def __init__(self):
        self.breaker = CircuitBreaker()

    async def call_async(self, model, prompt):
        raise NotImplementedError
EOF

cat > $PKG/providers/ollama_provider.py <<EOF
import requests
import time
from .base import BaseProvider

class OllamaProvider(BaseProvider):
    BASE_URL = "http://localhost:11434"

    async def call_async(self, model, prompt):
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

# =========================================================
# GOVERNANCE
# =========================================================

cat > $PKG/governance/policy_engine.py <<EOF
class PolicyEngine:
    def __init__(self):
        self.usage = {}

    def validate(self, tenant_id, tokens, cost):
        return True
EOF

# =========================================================
# OPTIMIZATION
# =========================================================

cat > $PKG/optimization/roi_engine.py <<EOF
class ROIEngine:
    def compute(self, cost, business_value):
        if cost == 0:
            return business_value
        return business_value / cost
EOF

# =========================================================
# BILLING
# =========================================================

cat > $PKG/billing/stripe_engine.py <<EOF
import stripe
from ..config import Config

class StripeEngine:
    def __init__(self):
        stripe.api_key = Config.STRIPE_API_KEY

    def meter(self, tenant_id, tokens):
        # Placeholder for metered billing
        pass
EOF

# =========================================================
# OBSERVABILITY
# =========================================================

cat > $PKG/observability/logging.py <<EOF
import logging

def get_logger():
    logger = logging.getLogger("ai_cost_controller")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
EOF

# =========================================================
# PERSISTENCE
# =========================================================

cat > $PKG/persistence/models.py <<EOF
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class LLMLog(Base):
    __tablename__ = "llm_logs"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String)
    provider = Column(String)
    model = Column(String)
    tokens = Column(Integer)
    cost = Column(Float)
EOF

# =========================================================
# ROOT FILES
# =========================================================

cat > requirements.txt <<EOF
groq
requests
sqlalchemy
stripe
redis
confluent-kafka
build
EOF

cat > setup.py <<EOF
from setuptools import setup, find_packages

setup(
    name="$PKG",
    version="$VERSION",
    packages=find_packages(),
    install_requires=[
        "groq",
        "requests",
        "sqlalchemy",
        "stripe",
        "redis",
        "confluent-kafka"
    ],
    python_requires=">=3.9",
)
EOF

cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
EOF

echo "📦 Building package..."
pip install build
python3 -m build
pip install dist/*.whl

echo "✅ Enterprise AI Cost Controller v2 installed successfully!"
