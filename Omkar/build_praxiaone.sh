#!/bin/bash
set -e

echo "==================================================="
echo "  PRAXIAONE ENTERPRISE AI - SYSTEM BOOTSTRAP"
echo "==================================================="

echo "[1/4] Creating base directories..."
mkdir -p ai_cost_controller/django_models/migrations
mkdir -p ai_cost_controller/management/commands
mkdir -p ai_cost_controller/providers
mkdir -p ai_cost_controller/templates/admin/ai_cost_controller/aiusagelog
mkdir -p ai_cost_controller/utils
mkdir -p ai_cost_migrations
mkdir -p praxiaone
mkdir -p templates

echo "[2/4] Generating Package Metadata & Configuration..."

cat << 'EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai_cost_controller"
version = "1.0.0"
description = "Enterprise AI Cost Management & Routing Engine"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "django>=4.2",
    "playwright>=1.40.0",
]

[tool.setuptools.packages.find]
include = ["ai_cost_controller*"]
EOF

cat << 'EOF' > README.md
# Praxiaone AI Cost Controller
Enterprise Token Economics and intelligent routing layer for Django.
EOF

cat << 'EOF' > requirements.txt
django>=4.2
playwright>=1.40.0
requests>=2.31.0
build
wheel
EOF

echo "[3/4] Injecting Core Python Files..."

# --- ai_cost_controller ---
touch ai_cost_controller/__init__.py
touch ai_cost_migrations/__init__.py

cat << 'EOF' > ai_cost_controller/settings.py
INSTALLED_APPS = ['ai_cost_controller']
EOF

cat << 'EOF' > ai_cost_controller/cost_logger.py
import logging
from django.utils import timezone
from django.db.models import Sum
from .django_models.models import AIUsageLog, UserBudget

logger = logging.getLogger(__name__)

class CostLogger:
    ALERT_THRESHOLD_USD = 0.50
    SLA_LATENCY_MAX_MS = 5000
    WARNING_BUDGET_PCT = 0.90

    @staticmethod
    def log(user, model, metrics, prompt_text="", response_text=""):
        try:
            in_tok = metrics.get('in', 0)
            out_tok = metrics.get('out', 0)
            latency = metrics.get('latency', 0)
            
            cost = float((in_tok * (float(model.input_cost_1m) / 1000000)) + \
                         (out_tok * (float(model.output_cost_1m) / 1000000)))
            
            AIUsageLog.objects.create(
                user=user,
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost=cost,
                latency_ms=latency,
                roi_score=metrics.get('roi_score', 0.00),
                prompt_text=prompt_text,
                response_text=response_text
            )
            
            if cost >= CostLogger.ALERT_THRESHOLD_USD:
                logger.critical(f"🚨 ANOMALY: User {user.username} spent ${cost:.4f} on a single {model.model_id} prompt!")

            if latency > CostLogger.SLA_LATENCY_MAX_MS:
                logger.warning(f"🐌 SLA BREACH: {model.provider.name}'s {model.model_id} took {latency}ms to respond. Consider penalizing this model in the router.")

            try:
                budget = UserBudget.objects.get(user=user)
                daily_cap = float(budget.daily_budget_cap)
                today = timezone.now().date()
                total_today = AIUsageLog.objects.filter(
                    user=user, timestamp__date=today
                ).aggregate(Sum('cost'))['cost__sum'] or 0.0
                
                if float(total_today) >= (daily_cap * CostLogger.WARNING_BUDGET_PCT):
                    logger.warning(f"⚠️ BUDGET WARNING: {user.username} has consumed {CostLogger.WARNING_BUDGET_PCT * 100}% of their daily cap (${total_today:.2f} / ${daily_cap:.2f}).")
            except UserBudget.DoesNotExist:
                pass
        except Exception as e:
            logger.error(f"Failed to securely save AI log to database: {e}")
EOF

cat << 'EOF' > ai_cost_controller/model_switcher.py
import logging
from .django_models.models import ModelPricing
from .roi_engine import ROIEngine

logger = logging.getLogger(__name__)

class ModelSwitcher:
    @staticmethod
    def get_routing_chain(prompt_text=""):
        active_models = list(ModelPricing.objects.filter(provider__is_active=True))
        if not active_models:
            raise ValueError("No active models configured in the database.")
        if not prompt_text:
            return sorted(active_models, key=lambda m: m.input_cost_1m, reverse=True)

        estimated_tokens = len(prompt_text.split()) * 1.3
        scored_models = []
        for model in active_models:
            if estimated_tokens > 8000 and model.provider.name.lower() == 'ollama':
                logger.info(f"Skipping {model.model_id}: Prompt too large for local fallback.")
                continue
            projected_roi = ROIEngine.calculate_projected_roi(model, prompt_text)
            if ROIEngine.analyze_prompt_intent(prompt_text) == 'general' and model.tier == 'cheap':
                projected_roi += 50.0  
            scored_models.append((projected_roi, model))

        scored_models.sort(key=lambda x: x[0], reverse=True)
        chain = [model for roi, model in scored_models]
        local_models = [m for m in chain if m.provider.name.lower() == 'ollama']
        cloud_models = [m for m in chain if m.provider.name.lower() != 'ollama']
        return cloud_models + local_models
EOF

cat << 'EOF' > ai_cost_controller/policy_engine.py
import re
import logging
from django.core.exceptions import PermissionDenied
from django.utils import timezone
from django.db.models import Sum
from .django_models.models import AIUsageLog, UserBudget

logger = logging.getLogger(__name__)

class EnterprisePolicyEngine:
    GLOBAL_DAILY_CAP = 1000.00 

    @classmethod
    def validate_request(cls, user, project_id=None):
        today = timezone.now().date()
        try:
            budget = UserBudget.objects.get(user=user)
            user_daily_cap = float(budget.daily_budget_cap)
        except UserBudget.DoesNotExist:
            user_daily_cap = 5.00 

        user_spend = AIUsageLog.objects.filter(
            user=user, timestamp__date=today
        ).aggregate(Sum('cost'))['cost__sum'] or 0.00
        
        if float(user_spend) >= user_daily_cap:
            logger.warning(f"User {user.username} blocked: Exceeded daily budget of ${user_daily_cap}")
            raise PermissionDenied(f"BUDGET DEPLETED: You have exceeded your daily cap of ${user_daily_cap:.2f}")

        global_spend = AIUsageLog.objects.filter(
            timestamp__date=today
        ).aggregate(Sum('cost'))['cost__sum'] or 0.00
        
        if float(global_spend) >= cls.GLOBAL_DAILY_CAP:
            logger.critical("Global enterprise daily AI budget reached. All API calls halted.")
            raise PermissionDenied("CRITICAL: Enterprise global AI budget exhausted. Systems locked.")
            
        return True

    @classmethod
    def sanitize_prompt(cls, prompt_text):
        sanitized = prompt_text
        dlp_patterns = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[REDACTED_SSN]',
            r'\b(?:\d[ -]*?){13,16}\b': '[REDACTED_CREDIT_CARD]',
            r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+': '[REDACTED_EMAIL]',
            r'(?i)AKIA[0-9A-Z]{16}': '[REDACTED_AWS_KEY]' 
        }
        for pattern, replacement in dlp_patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized)
        if sanitized != prompt_text:
            logger.warning("DLP Engine intercepted and masked sensitive data before routing.")
        return sanitized

    @classmethod
    def enforce_rbac(cls, user, requested_model_id):
        premium_models = ['gpt-5.2', 'grok-4']
        if requested_model_id in premium_models and not user.is_staff:
            logger.warning(f"RBAC Violation: {user.username} attempted to access {requested_model_id}.")
            return "gemini-1.5-pro" 
        return requested_model_id
EOF

cat << 'EOF' > ai_cost_controller/roi_engine.py
import re

class ROIEngine:
    TASK_PROFILES = {
        'coding': {'speed_weight': 0.8, 'capability_weight': 1.5},
        'creative': {'speed_weight': 1.0, 'capability_weight': 1.2},
        'data_extraction': {'speed_weight': 1.5, 'capability_weight': 1.0},
        'general': {'speed_weight': 1.0, 'capability_weight': 1.0}
    }

    @staticmethod
    def analyze_prompt_intent(prompt_text):
        prompt_lower = prompt_text.lower()
        if re.search(r'\b(def|class|function|html|css|python|react|django|bug|error)\b', prompt_lower):
            return 'coding'
        elif re.search(r'\b(write|create|poem|story|imagine|blog|essay)\b', prompt_lower):
            return 'creative'
        elif re.search(r'\b(extract|summarize|json|parse|list|table)\b', prompt_lower):
            return 'data_extraction'
        return 'general'

    @staticmethod
    def calculate_projected_roi(model, prompt_text):
        task_type = ROIEngine.analyze_prompt_intent(prompt_text)
        profile = ROIEngine.TASK_PROFILES[task_type]
        in_cost = float(model.input_cost_1m) / 1000000 * 500
        out_cost = float(model.output_cost_1m) / 1000000 * 500
        projected_cost = in_cost + out_cost
        if projected_cost <= 0: projected_cost = 0.000001
        capability_score = 100 if model.tier == 'premium' else (50 if model.tier == 'mid' else 20)
        roi = (capability_score * profile['capability_weight']) / (projected_cost * profile['speed_weight'] * 10000)
        return round(min(roi, 99.99), 2)

    @staticmethod
    def calculate_transaction_value(model, latency_ms, in_tokens, out_tokens, prompt_text=""):
        task_type = ROIEngine.analyze_prompt_intent(prompt_text)
        profile = ROIEngine.TASK_PROFILES[task_type]
        in_cost = float(model.input_cost_1m) / 1000000 * in_tokens
        out_cost = float(model.output_cost_1m) / 1000000 * out_tokens
        total_cost = in_cost + out_cost
        if total_cost <= 0: total_cost = 0.000001
        speed_factor = 1000 / (latency_ms + 1)
        score = (speed_factor * profile['speed_weight']) / (total_cost * 100)
        return round(min(score, 99.99), 2)
EOF

cat << 'EOF' > ai_cost_controller/router.py
import logging
import time
import json
from .policy_engine import EnterprisePolicyEngine
from .model_switcher import ModelSwitcher
from .roi_engine import ROIEngine
from .cost_logger import CostLogger
from .providers.ollama_provider import OllamaProvider
from .django_models.models import ModelPricing

logger = logging.getLogger(__name__)

class AIRouterFacade:
    @staticmethod
    def get_provider_instance(provider_name):
        if provider_name.lower() == 'ollama':
            return OllamaProvider()
        return None

    @staticmethod
    def execute_prompt(user, task_type, prompt, project_id=None):
        pass

    @staticmethod
    def execute_prompt_stream(user, task_type, prompt, model_override="auto"):
        EnterprisePolicyEngine.validate_request(user, None)
        safe_prompt = EnterprisePolicyEngine.sanitize_prompt(prompt)

        if model_override and model_override != "auto":
            model_override = EnterprisePolicyEngine.enforce_rbac(user, model_override)

        if model_override and model_override != "auto":
            try:
                selected_db_model = ModelPricing.objects.get(model_id=model_override)
                model_chain = [selected_db_model]
            except ModelPricing.DoesNotExist:
                raise RuntimeError(f"Selected model {model_override} does not exist in database.")
        else:
            model_chain = ModelSwitcher.get_routing_chain(prompt_text=safe_prompt)
        
        selected_model = None
        provider_instance = None
        
        for model in model_chain:
            if model.provider.name.lower() != 'ollama':
                selected_model = model
                provider_instance = None
                break
            provider = AIRouterFacade.get_provider_instance(model.provider.name)
            if provider and hasattr(provider, 'invoke_stream'):
                selected_model = model
                provider_instance = provider
                break
                
        if not selected_model:
            raise RuntimeError("No models available for routing.")
            
        def stream_generator():
            full_text = ""
            metrics = None
            try:
                if provider_instance is None and selected_model.provider.name.lower() != 'ollama':
                    fake_sentence = f"Hello! This is a simulated AI generated response from {selected_model.provider.name}'s {selected_model.model_id}. I am acting as a placeholder because no API keys were provided. The Enterprise Router successfully routed to me and will now deduct costs from your budget!"
                    fake_words = fake_sentence.split(' ')
                    for word in fake_words:
                        chunk_text = word + " "
                        full_text += chunk_text
                        yield f"data: {json.dumps({'text': chunk_text})}\n\n"
                        time.sleep(0.05) 
                    metrics = {
                        "in": len(safe_prompt.split()) + 10, 
                        "out": len(fake_words),         
                        "latency": 0.85                 
                    }
                else:
                    for chunk_data in provider_instance.invoke_stream(selected_model.model_id, safe_prompt):
                        if "text" in chunk_data:
                            full_text += chunk_data["text"]
                            yield f"data: {json.dumps({'text': chunk_data['text']})}\n\n"
                        if "metrics" in chunk_data:
                            metrics = chunk_data["metrics"]
                
                if metrics:
                    roi_score = ROIEngine.calculate_transaction_value(
                        selected_model, metrics.get('latency', 0), metrics.get('in', 0), metrics.get('out', 0), safe_prompt
                    )
                    metrics['roi_score'] = roi_score
                    CostLogger.log(user, selected_model, metrics, prompt_text=safe_prompt, response_text=full_text)
                
                yield f"data: {json.dumps({'done': True, 'model': selected_model.model_id})}\n\n"
            except Exception as e:
                logger.error(f"Streaming failed mid-generation: {e}")
                error_msg = f"\n\n[System Warning: Connection to {selected_model.model_id} failed. Error: {str(e)}]\n"
                yield f"data: {json.dumps({'text': error_msg})}\n\n"
                yield f"data: {json.dumps({'done': True, 'model': 'Failed'})}\n\n"

        return stream_generator()
EOF

# --- django_models ---
touch ai_cost_controller/django_models/__init__.py
cat << 'EOF' > ai_cost_controller/django_models/models.py
from django.db import models
from django.contrib.auth.models import User

class Provider(models.Model):
    name = models.CharField(max_length=50, unique=True)
    is_active = models.BooleanField(default=True)
    def __str__(self): return self.name

class ModelPricing(models.Model):
    provider = models.ForeignKey(Provider, on_delete=models.CASCADE)
    model_id = models.CharField(max_length=100, unique=True)
    input_cost_1m = models.DecimalField(max_digits=10, decimal_places=4)
    output_cost_1m = models.DecimalField(max_digits=10, decimal_places=4)
    tier = models.CharField(max_length=20, choices=[('cheap', 'Cheap'), ('mid', 'Mid'), ('premium', 'Premium')])
    def __str__(self): return self.model_id

class AIUsageLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(ModelPricing, on_delete=models.CASCADE)
    input_tokens = models.IntegerField()
    output_tokens = models.IntegerField()
    cost = models.DecimalField(max_digits=12, decimal_places=6)
    latency_ms = models.IntegerField()
    roi_score = models.DecimalField(max_digits=6, decimal_places=2, default=0.00)
    prompt_text = models.TextField(blank=True, null=True)
    response_text = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

class UserBudget(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    daily_budget_cap = models.DecimalField(max_digits=10, decimal_places=4, default=5.00)
    monthly_budget_cap = models.DecimalField(max_digits=10, decimal_places=4, default=50.00)
    def __str__(self): 
        return f"{self.user.username} - Budget"
EOF

cat << 'EOF' > ai_cost_controller/django_models/admin.py
from django.contrib import admin
from django.db.models import Sum
from .models import Provider, ModelPricing, AIUsageLog, UserBudget

@admin.register(AIUsageLog)
class AIUsageLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'user', 'model', 'cost', 'latency_ms', 'roi_score')
    list_filter = ('model__provider', 'user', 'timestamp')
    change_list_template = "admin/ai_cost_controller/aiusagelog/change_list.html"
    
    def changelist_view(self, request, extra_context=None):
        aggregates = AIUsageLog.objects.aggregate(
            total_spend=Sum('cost'),
            total_input=Sum('input_tokens'),
            total_output=Sum('output_tokens')
        )
        extra_context = extra_context or {}
        extra_context['summary_spend'] = aggregates['total_spend'] or 0.00
        extra_context['summary_tokens'] = (aggregates['total_input'] or 0) + (aggregates['total_output'] or 0)
        return super().changelist_view(request, extra_context=extra_context)

admin.site.register(Provider)
admin.site.register(ModelPricing)

@admin.register(UserBudget)
class UserBudgetAdmin(admin.ModelAdmin):
    list_display = ('user', 'daily_budget_cap', 'monthly_budget_cap')
    search_fields = ('user__username',)
EOF

# --- management ---
touch ai_cost_controller/management/__init__.py
touch ai_cost_controller/management/commands/__init__.py
cat << 'EOF' > ai_cost_controller/management/commands/sync_ai_prices.py
from django.core.management.base import BaseCommand
from ai_cost_controller.utils.token_counter import PricingCrawler
from ai_cost_controller.django_models.models import Provider, ModelPricing

class Command(BaseCommand):
    help = 'Crawls and synchronizes the latest AI API prices.'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting price synchronization...")
        pricing_data = PricingCrawler.extract_prices()
        for prov_name, models in pricing_data.items():
            provider, _ = Provider.objects.get_or_create(name=prov_name)
            for m in models:
                ModelPricing.objects.update_or_create(
                    provider=provider,
                    model_id=m['id'],
                    defaults={
                        'input_cost_1m': m['in_1m'],
                        'output_cost_1m': m['out_1m'],
                        'tier': m['tier']
                    }
                )
                self.stdout.write(f"Synced: {m['id']} via {prov_name}")
        self.stdout.write(self.style.SUCCESS("Successfully updated all DB prices!"))
EOF

# --- providers ---
touch ai_cost_controller/providers/__init__.py
cat << 'EOF' > ai_cost_controller/providers/anthropic_provider.py
import os, time
class AnthropicProvider:
    def invoke(self, model_name, prompt):
        return {"text": f"[MOCK ANTHROPIC] {model_name} executed.", "metrics": {"in": 120, "out": 250, "latency": 600}}
EOF
cat << 'EOF' > ai_cost_controller/providers/ollama_provider.py
import requests, time, json
class OllamaProvider:
    def invoke_stream(self, model_name, prompt):
        start = time.time()
        try:
            res = requests.post(
                "http://localhost:11434/api/generate", 
                json={"model": model_name, "prompt": prompt, "stream": True}, 
                stream=True
            )
            res.raise_for_status()
            for line in res.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield {"text": chunk.get("response", "")}
                    if chunk.get("done"):
                        latency = int((time.time() - start) * 1000)
                        yield {"metrics": {"in": chunk.get("prompt_eval_count", 0), "out": chunk.get("eval_count", 0), "latency": latency}}
        except Exception as e:
            raise RuntimeError(f"Ollama streaming connection failed: {e}")
EOF

# --- utils ---
touch ai_cost_controller/utils/__init__.py
cat << 'EOF' > ai_cost_controller/utils/token_counter.py
from playwright.sync_api import sync_playwright
import re

class PricingCrawler:
    @staticmethod
    def clean_price(price_string):
        if not price_string: return 0.00
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", price_string)
        return float(numbers[0]) if numbers else 0.00

    @staticmethod
    def extract_prices():
        pricing_data = {
            "OpenAI": [], "Anthropic": [], "Google": [], "xAI": [],
            "Ollama": [{"id": "llama3-local", "in_1m": 0.00, "out_1m": 0.00, "tier": "cheap"}]
        }
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto("https://openai.com/api/pricing/", timeout=15000)
                gpt52_in = page.locator(r"#main > div > section:nth-child(2) > div > div > div.\@container\/pricing-card.w-full.max-w-container.grid.grid-cols-12 > div > div > div:nth-child(1) > div.mt-auto > div > div:nth-child(2) > div").inner_text()   
                gpt52_out = page.locator(r"#main > div > section:nth-child(2) > div > div > div.\@container\/pricing-card.w-full.max-w-container.grid.grid-cols-12 > div > div > div:nth-child(1) > div.mt-auto > div > div:nth-child(4) > div").inner_text() 
                pricing_data["OpenAI"].append({"id": "gpt-5.2", "in_1m": PricingCrawler.clean_price(gpt52_in), "out_1m": PricingCrawler.clean_price(gpt52_out), "tier": "premium"})
            except Exception as e:
                pricing_data["OpenAI"] = [{"id": "gpt-5.2", "in_1m": 1.75, "out_1m": 14.00, "tier": "premium"}]
                pricing_data["Anthropic"] = [{"id": "claude-4-5-sonnet", "in_1m": 3.00, "out_1m": 15.00, "tier": "mid"}]
                pricing_data["Google"] = [{"id": "gemini-3-pro", "in_1m": 1.25, "out_1m": 5.00, "tier": "premium"}]
                pricing_data["xAI"] = [{"id": "grok-4", "in_1m": 2.00, "out_1m": 10.00, "tier": "mid"}]
            finally:
                browser.close()
        return pricing_data
EOF

# --- praxiaone & templates ---
touch praxiaone/__init__.py
cat << 'EOF' > praxiaone/settings.py
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'django-insecure-br_@ybn4#e@_om*gn^dvvyz_izofs*v*_^!w#^ei_s5-)yc5e6'
DEBUG = True
ALLOWED_HOSTS = []
INSTALLED_APPS = [
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
    'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
    'ai_cost_controller', 'ai_cost_controller.django_models',
]
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware', 'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware', 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware', 'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
ROOT_URLCONF = 'praxiaone.urls'
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request', 'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
WSGI_APPLICATION = 'praxiaone.wsgi.application'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True
STATIC_URL = 'static/'
MIGRATION_MODULES = { 'ai_cost_controller': 'ai_cost_migrations' }
EOF

cat << 'EOF' > praxiaone/wsgi.py
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'praxiaone.settings')
application = get_wsgi_application()
EOF

cat << 'EOF' > praxiaone/urls.py
from django.contrib import admin
from django.urls import path
from .views import test_ai_route, dashboard_data_api, dashboard_ui

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/test-ai/', test_ai_route, name='test_ai_route'), 
    path('api/dashboard-data/', dashboard_data_api, name='dashboard_data_api'), 
    path('dashboard/', dashboard_ui, name='dashboard_ui'),
]
EOF

cat << 'EOF' > praxiaone/views.py
import json
from django.http import StreamingHttpResponse, JsonResponse
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Sum
from django.shortcuts import render
from ai_cost_controller.router import AIRouterFacade
from ai_cost_controller.django_models.models import AIUsageLog, UserBudget, ModelPricing

@csrf_exempt
def test_ai_route(request):
    prompt = ""
    model_override = "auto"
    username = "omkar"
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt', '')
            model_override = data.get('model', 'auto') 
            username = data.get('user', 'omkar')       
        except json.JSONDecodeError: return JsonResponse({"error": "Invalid JSON."}, status=400)
    if not prompt: return JsonResponse({"error": "No prompt provided."}, status=400)
    user = User.objects.filter(username=username).first()
    if not user: return JsonResponse({"error": "No user found in database."}, status=400)
    try:
        stream_generator = AIRouterFacade.execute_prompt_stream(user, 'reasoning', prompt, model_override)
        response = StreamingHttpResponse(stream_generator, content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no' 
        return response
    except Exception as e: return JsonResponse({"status": "failed", "error": str(e)}, status=500)

def dashboard_data_api(request):
    username = request.GET.get('user', 'omkar') 
    today = timezone.now().date()
    try:
        budget = UserBudget.objects.get(user__username=username)
        daily_cap, monthly_cap = float(budget.daily_budget_cap), float(budget.monthly_budget_cap)
    except UserBudget.DoesNotExist:
        daily_cap, monthly_cap = 5.00, 50.00
    
    logs_today = AIUsageLog.objects.filter(user__username=username, timestamp__date=today)
    cost_today = float(logs_today.aggregate(Sum('cost'))['cost__sum'] or 0.0)
    in_tokens = logs_today.aggregate(Sum('input_tokens'))['input_tokens__sum'] or 0
    out_tokens = logs_today.aggregate(Sum('output_tokens'))['output_tokens__sum'] or 0
    remaining_daily = max(0.0, daily_cap - cost_today)
    
    logs_month = AIUsageLog.objects.filter(user__username=username, timestamp__month=today.month, timestamp__year=today.year)
    cost_month = float(logs_month.aggregate(Sum('cost'))['cost__sum'] or 0.0)
    remaining_monthly = max(0.0, monthly_cap - cost_month)
    
    chart_data, stats_map = {"labels": [], "costs": []}, {}
    model_stats = logs_today.values('model__model_id').annotate(total_cost=Sum('cost'), in_tok=Sum('input_tokens'), out_tok=Sum('output_tokens'))
    for stat in model_stats:
        label = stat['model__model_id'] or 'Unknown' 
        chart_data["labels"].append(label)
        chart_data["costs"].append(float(stat['total_cost']))
        stats_map[label] = stat['in_tok'] + stat['out_tok']
        
    models = ModelPricing.objects.all()
    available_models = [{"id": m.model_id, "provider": m.provider.name, "in_cost": float(m.input_cost_1m), "out_cost": float(m.output_cost_1m), "tokens_today": stats_map.get(m.model_id, 0)} for m in models]
    return JsonResponse({"budget": {"daily_cap": daily_cap, "monthly_cap": monthly_cap, "total_cost_today": cost_today, "remaining_budget": remaining_daily, "remaining_monthly": remaining_monthly, "total_tokens_today": in_tokens+out_tokens}, "chart_data": chart_data, "available_models": available_models})

def dashboard_ui(request): return render(request, 'dashboard.html')
EOF

cat << 'EOF' > templates/dashboard.html
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <title>Praxiaone AI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 p-8 flex flex-col items-center justify-center h-screen">
    <h1 class="text-4xl text-blue-400 font-bold mb-4">Dashboard Built Successfully</h1>
    <p class="text-gray-400">The server is running. Go back to your code to expand the UI!</p>
</body>
</html>
EOF

cat << 'EOF' > manage.py
#!/usr/bin/env python
import os, sys
def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'praxiaone.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
if __name__ == '__main__':
    main()
EOF

echo "[4/4] Building Wheel and Bootstrapping Server..."

# Build Environment & Wheel
python -m venv venv
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install build wheel
python -m build

# Install the generated wheel
echo "Installing the compiled AI Cost Controller .whl..."
pip install dist/*.whl

# Django Setup
echo "Applying database migrations..."
python manage.py makemigrations ai_cost_controller
python manage.py makemigrations
python manage.py migrate

echo "Creating Superuser 'omkar' with password 'admin'..."
python manage.py shell -c "from django.contrib.auth.models import User; User.objects.create_superuser('omkar', 'omkar@example.com', 'admin')" || true

echo "==================================================="
echo "  System Ready! "
echo "  Run: python manage.py runserver"
echo "==================================================="
