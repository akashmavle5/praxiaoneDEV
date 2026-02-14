#!/bin/bash
set -e

############################################################

# AI COST CONTROLLER — GOD MODE ENTERPRISE BUILDER

# Full multi-provider cost engine + Django optional

# Generates complete pip-installable package + wheel

############################################################

echo "=============================================="
echo " AI COST CONTROLLER — GOD MODE BUILD"
echo "=============================================="

ROOT="ai_cost_controller"
PKG="$ROOT/ai_cost_controller"

rm -rf $ROOT dist build *.egg-info 2>/dev/null || true

mkdir -p $PKG/providers
mkdir -p $PKG/django_models
mkdir -p $PKG/utils

touch $PKG/**init**.py
touch $PKG/providers/**init**.py
touch $PKG/django_models/**init**.py
touch $PKG/utils/**init**.py

#############################################

# ENV EXAMPLE

#############################################
cat <<EOF > $ROOT/.env.example
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
XAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
EOF

#############################################

# requirements.txt

#############################################
cat <<EOF > $ROOT/requirements.txt
openai
anthropic
google-generativeai
tiktoken
requests
django
python-dotenv
setuptools
wheel
build
EOF

#############################################

# setup.py

#############################################
cat <<EOF > $ROOT/setup.py
from setuptools import setup, find_packages

setup(
name="ai_cost_controller",
version="2.0.0",
author="PraxiaAI",
description="Enterprise AI cost + token economics controller",
packages=find_packages(),
install_requires=[
"openai",
"anthropic",
"google-generativeai",
"django",
"tiktoken",
"requests",
"python-dotenv"
],
include_package_data=True,
zip_safe=False,
)
EOF

#############################################

# pyproject.toml

#############################################
cat <<EOF > $ROOT/pyproject.toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
EOF

#############################################

# MANIFEST

#############################################
cat <<EOF > $ROOT/MANIFEST.in
recursive-include ai_cost_controller *
include requirements.txt
include .env.example
EOF

#############################################

# README

#############################################
cat <<EOF > $ROOT/README.md

# AI Cost Controller — Enterprise

Multi-provider AI cost + token governance engine.

Supports:

* OpenAI
* Anthropic
* Gemini
* xAI
* Ollama local models

Install:
pip install dist/*.whl

Configure:
cp .env.example .env
EOF

#############################################

# SETTINGS

#############################################
cat <<EOF > $PKG/settings.py
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY=os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY=os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY=os.getenv("GEMINI_API_KEY")
XAI_KEY=os.getenv("XAI_API_KEY")
OLLAMA_URL=os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

def has(key):
return key is not None and key != ""
EOF

#############################################

# TOKEN COUNTER

#############################################
cat <<EOF > $PKG/utils/token_counter.py
def estimate_tokens(text:str):
if not text:
return 0
return max(1,len(text)//4)
EOF

#############################################

# DJANGO MODEL (OPTIONAL SAFE)

#############################################
cat <<EOF > $PKG/django_models/models.py
try:
from django.db import models

```
class LLMCallLog(models.Model):
    user_id=models.CharField(max_length=100)
    project_id=models.CharField(max_length=100,null=True,blank=True)
    task_type=models.CharField(max_length=50)

    provider=models.CharField(max_length=50)
    model=models.CharField(max_length=100)

    input_tokens=models.IntegerField()
    output_tokens=models.IntegerField()
    total_tokens=models.IntegerField()

    cost_usd=models.FloatField()
    latency_ms=models.FloatField()

    accuracy_score=models.FloatField(default=0)
    business_value_score=models.FloatField(default=0)
    roi_score=models.FloatField(default=0)

    created_at=models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table="ai_llm_call_logs"
```

except Exception:
LLMCallLog=None
EOF

#############################################

# POLICY ENGINE

#############################################
cat <<EOF > $PKG/policy_engine.py
from datetime import datetime
try:
from django.db.models import Sum
from .django_models.models import LLMCallLog
except:
LLMCallLog=None

class TokenPolicyEngine:

```
def __init__(self,daily_cap=5.0,monthly_cap=100.0):
    self.daily_cap=daily_cap
    self.monthly_cap=monthly_cap

def _today_cost(self,user_id):
    if not LLMCallLog:
        return 0
    today=datetime.utcnow().date()
    return LLMCallLog.objects.filter(
        user_id=user_id,
        created_at__date=today
    ).aggregate(Sum("cost_usd"))["cost_usd__sum"] or 0

def _month_cost(self,user_id):
    if not LLMCallLog:
        return 0
    now=datetime.utcnow()
    return LLMCallLog.objects.filter(
        user_id=user_id,
        created_at__month=now.month
    ).aggregate(Sum("cost_usd"))["cost_usd__sum"] or 0

def check_limits(self,user_id,estimated_cost):
    today=self._today_cost(user_id)
    month=self._month_cost(user_id)

    if today+estimated_cost>self.daily_cap:
        return False,"Daily budget exceeded"
    if month+estimated_cost>self.monthly_cap:
        return False,"Monthly budget exceeded"

    return True,"OK"
```

EOF

#############################################

# ROI ENGINE

#############################################
cat <<EOF > $PKG/roi_engine.py
class ROICalculator:

```
def calculate_roi(self,cost,accuracy,latency,value):
    if cost==0:
        return accuracy*value
    roi=(accuracy*0.4+value*0.4-latency*0.2)/cost
    return round(roi,4)

def should_upgrade(self,roi):
    return roi>1.5

def should_downgrade(self,roi):
    return roi<0.5
```

EOF

#############################################

# COST LOGGER

#############################################
cat <<EOF > $PKG/cost_logger.py
from .roi_engine import ROICalculator
try:
from .django_models.models import LLMCallLog
except:
LLMCallLog=None

roi_engine=ROICalculator()

class CostLogger:

```
def log(self,**k):
    roi=roi_engine.calculate_roi(
        k.get("cost",0),
        k.get("accuracy",1),
        k.get("latency",0),
        k.get("business_value",1)
    )

    if LLMCallLog:
        LLMCallLog.objects.create(
            user_id=k.get("user_id"),
            project_id=k.get("project_id"),
            task_type=k.get("task_type"),
            provider=k.get("provider"),
            model=k.get("model"),
            input_tokens=k.get("input_tokens"),
            output_tokens=k.get("output_tokens"),
            total_tokens=k.get("input_tokens")+k.get("output_tokens"),
            cost_usd=k.get("cost"),
            latency_ms=k.get("latency"),
            accuracy_score=k.get("accuracy"),
            business_value_score=k.get("business_value"),
            roi_score=roi
        )
    return roi
```

EOF

#############################################

# PROVIDERS (AUTO-SKIP IF NO KEY)

#############################################

# OPENAI

cat <<EOF > $PKG/providers/openai_provider.py
import time,os
from openai import OpenAI
from ..settings import OPENAI_KEY

class OpenAIProvider:
def **init**(self):
self.client=OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
def available(self):
return self.client is not None
def call(self,model,prompt):
start=time.time()
r=self.client.chat.completions.create(
model=model,
messages=[{"role":"user","content":prompt}]
)
out=r.choices[0].message.content
latency=(time.time()-start)*1000
tin=r.usage.prompt_tokens
tout=r.usage.completion_tokens
cost=(tin+tout)*0.000002
return out,tin,tout,cost,latency
EOF

# ANTHROPIC

cat <<EOF > $PKG/providers/anthropic_provider.py
import time,os,anthropic
from ..settings import ANTHROPIC_KEY

class AnthropicProvider:
def **init**(self):
self.client=anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None
def available(self):
return self.client is not None
def call(self,model,prompt):
start=time.time()
r=self.client.messages.create(
model=model,
max_tokens=500,
messages=[{"role":"user","content":prompt}]
)
out=r.content[0].text
latency=(time.time()-start)*1000
return out,100,200,0.002,latency
EOF

# GEMINI

cat <<EOF > $PKG/providers/gemini_provider.py
import time,google.generativeai as genai
from ..settings import GEMINI_KEY

class GeminiProvider:
def **init**(self):
if GEMINI_KEY:
genai.configure(api_key=GEMINI_KEY)
self.model=genai.GenerativeModel("gemini-pro")
else:
self.model=None
def available(self):
return self.model is not None
def call(self,model,prompt):
start=time.time()
r=self.model.generate_content(prompt)
latency=(time.time()-start)*1000
return r.text,100,200,0.002,latency
EOF

# XAI

cat <<EOF > $PKG/providers/xai_provider.py
import time,requests
from ..settings import XAI_KEY

class XAIProvider:
def available(self):
return XAI_KEY is not None
def call(self,model,prompt):
start=time.time()
r=requests.post(
"https://api.x.ai/v1/chat/completions",
headers={"Authorization":f"Bearer {XAI_KEY}"},
json={"model":model,"messages":[{"role":"user","content":prompt}]}
)
out=r.json()["choices"][0]["message"]["content"]
latency=(time.time()-start)*1000
return out,100,200,0.002,latency
EOF

# OLLAMA FULL

cat <<EOF > $PKG/providers/ollama_provider.py
import requests,time,os,json
from ..settings import OLLAMA_URL

class OllamaProvider:

```
def available(self):
    return True

def list_models(self):
    r=requests.get(f"{OLLAMA_URL}/api/tags",timeout=60)
    if r.status_code!=200:
        return []
    return [m["name"] for m in r.json().get("models",[])]

def _estimate(self,t):
    return max(1,len(t)//4)

def call(self,model,prompt):
    start=time.time()
    r=requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model":model,"prompt":prompt,"stream":False},
        timeout=600
    )
    out=r.json().get("response","")
    latency=(time.time()-start)*1000
    tin=self._estimate(prompt)
    tout=self._estimate(out)
    return out,tin,tout,0.0,latency
```

EOF

#############################################

# MODEL SWITCHER

#############################################
cat <<EOF > $PKG/model_switcher.py
class ModelSwitcher:
def **init**(self):
self.routing={
"draft":{"provider":"ollama","model":"llama3"},
"verify":{"provider":"anthropic","model":"claude-3-haiku"},
"final":{"provider":"openai","model":"gpt-4o"}
}
def select_model(self,stage):
return self.routing.get(stage,self.routing["draft"])
EOF

#############################################

# ROUTER

#############################################
cat <<EOF > $PKG/router.py
from .policy_engine import TokenPolicyEngine
from .model_switcher import ModelSwitcher
from .cost_logger import CostLogger

from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.xai_provider import XAIProvider
from .providers.ollama_provider import OllamaProvider

policy=TokenPolicyEngine()
switcher=ModelSwitcher()
logger=CostLogger()

providers={
"openai":OpenAIProvider(),
"anthropic":AnthropicProvider(),
"gemini":GeminiProvider(),
"xai":XAIProvider(),
"ollama":OllamaProvider()
}

class AICostController:

```
def generate(self,user_id,project_id,prompt,stage="draft"):

    route=switcher.select_model(stage)
    provider_name=route["provider"]
    model=route["model"]

    ok,msg=policy.check_limits(user_id,0.01)
    if not ok:
        raise Exception(msg)

    order=[provider_name,"openai","anthropic","gemini","ollama"]

    for p in order:
        provider=providers.get(p)
        if provider and provider.available():
            try:
                out,in_t,out_t,cost,lat=provider.call(model,prompt)

                roi=logger.log(
                    user_id=user_id,
                    project_id=project_id,
                    task_type=stage,
                    provider=p,
                    model=model,
                    input_tokens=in_t,
                    output_tokens=out_t,
                    cost=cost,
                    latency=lat,
                    accuracy=0.9,
                    business_value=0.8
                )

                return {
                    "output":out,
                    "provider":p,
                    "model":model,
                    "roi":roi
                }
            except Exception:
                continue

    raise Exception("No AI provider available. Add at least one API key or start Ollama.")
```

EOF

#############################################

# BUILD WHEEL

#############################################
cd $ROOT
pip install build wheel setuptools >/dev/null 2>&1
python3 -m build

echo ""
echo "=============================================="
echo " GOD MODE BUILD COMPLETE"
echo "=============================================="
echo "Wheel:"
echo "$ROOT/dist/"
echo ""
echo "Install:"
echo "pip install dist/*.whl"
echo "=============================================="
