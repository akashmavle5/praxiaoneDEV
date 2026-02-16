import os
import shutil
import subprocess
from pathlib import Path

print("==============================================")
print("AI COST CONTROLLER — WINDOWS GOD MODE BUILD")
print("==============================================")

ROOT = Path("ai_cost_controller")
PKG = ROOT / "ai_cost_controller"

# clean old
if ROOT.exists():
    shutil.rmtree(ROOT)

if Path("dist").exists():
    shutil.rmtree("dist")

if Path("build").exists():
    shutil.rmtree("build")

# create structure
(PKG / "providers").mkdir(parents=True, exist_ok=True)
(PKG / "django_models").mkdir(parents=True, exist_ok=True)
(PKG / "utils").mkdir(parents=True, exist_ok=True)

(PKG / "__init__.py").touch()
(PKG / "providers/__init__.py").touch()
(PKG / "django_models/__init__.py").touch()
(PKG / "utils/__init__.py").touch()

#############################################
# ENV
#############################################
(ROOT / ".env.example").write_text(
"""OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
XAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
"""
)

#############################################
# requirements
#############################################
(ROOT / "requirements.txt").write_text(
"""openai
anthropic
google-generativeai
tiktoken
requests
django
python-dotenv
setuptools
wheel
build
"""
)

#############################################
# setup.py
#############################################
(ROOT / "setup.py").write_text(
"""from setuptools import setup, find_packages

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
"""
)

#############################################
# pyproject
#############################################
(ROOT / "pyproject.toml").write_text(
"""[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
"""
)

#############################################
# MANIFEST
#############################################
(ROOT / "MANIFEST.in").write_text(
"""recursive-include ai_cost_controller *
include requirements.txt
include .env.example
"""
)

#############################################
# settings
#############################################
(PKG / "settings.py").write_text(
"""import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY=os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY=os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY=os.getenv("GEMINI_API_KEY")
XAI_KEY=os.getenv("XAI_API_KEY")
OLLAMA_URL=os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

def has(key):
    return key is not None and key!=""
"""
)

#############################################
# token counter
#############################################
(PKG / "utils/token_counter.py").write_text(
"""def estimate_tokens(text:str):
    if not text:
        return 0
    return max(1,len(text)//4)
"""
)

#############################################
# django model optional
#############################################
(PKG / "django_models/models.py").write_text(
"""try:
    from django.db import models

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
except:
    LLMCallLog=None
"""
)

#############################################
# policy engine
#############################################
(PKG / "policy_engine.py").write_text(
"""from datetime import datetime
try:
    from django.db.models import Sum
    from .django_models.models import LLMCallLog
except:
    LLMCallLog=None

class TokenPolicyEngine:

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
"""
)

#############################################
# simple router + cost logger etc
#############################################
(PKG / "roi_engine.py").write_text(
"""class ROICalculator:

    def calculate_roi(self,cost,accuracy,latency,value):
        if cost==0:
            return accuracy*value
        roi=(accuracy*0.4+value*0.4-latency*0.2)/cost
        return round(roi,4)
"""
)

(PKG / "cost_logger.py").write_text(
"""from .roi_engine import ROICalculator
roi_engine=ROICalculator()

class CostLogger:
    def log(self,**k):
        return roi_engine.calculate_roi(
            k.get("cost",0),
            k.get("accuracy",1),
            k.get("latency",0),
            k.get("business_value",1)
        )
"""
)

(PKG / "model_switcher.py").write_text(
"""class ModelSwitcher:
    def __init__(self):
        self.routing={
            "draft":{"provider":"ollama","model":"llama3"},
            "verify":{"provider":"anthropic","model":"claude-3-haiku"},
            "final":{"provider":"openai","model":"gpt-4o"}
        }
    def select_model(self,stage):
        return self.routing.get(stage,self.routing["draft"])
"""
)

(PKG / "router.py").write_text(
"""class AICostController:
    def generate(self,user_id,project_id,prompt,stage="draft"):
        return {"output":"AI controller ready","stage":stage}
"""
)

#############################################
# BUILD
#############################################
print("Installing build tools...")
subprocess.run(["pip","install","build","wheel","setuptools"],shell=True)

print("Building wheel...")
os.chdir(ROOT)
subprocess.run(["python","-m","build"],shell=True)

print("")
print("======================================")
print("BUILD COMPLETE")
print("Wheel inside dist/")
print("Install with:")
print("pip install dist/*.whl")
print("======================================")
