from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ai_cost_controller",
    version="2.0.0",
    description="Enterprise AI orchestration, cost governance, hybrid routing and distributed control plane",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Your Name",
    author_email="your@email.com",

    url="https://github.com/yourusername/ai_cost_controller",
    project_urls={
        "Documentation": "https://github.com/yourusername/ai_cost_controller/docs",
        "Source": "https://github.com/yourusername/ai_cost_controller",
        "Issues": "https://github.com/yourusername/ai_cost_controller/issues",
    },

    packages=find_packages(include=["ai_cost_controller", "ai_cost_controller.*"]),
    include_package_data=True,

    python_requires=">=3.9",

    install_requires=[
        # Core providers
        "groq>=0.11.0,<1.0.0",
        "requests>=2.31.0,<3.0.0",

        # Database
        "sqlalchemy>=2.0.0,<3.0.0",

        # Billing
        "stripe>=8.0.0,<9.0.0",

        # Distributed infra
        "redis>=5.0.0,<6.0.0",
        "confluent-kafka>=2.3.0,<3.0.0",

        # Token accounting
        "tiktoken>=0.5.0,<1.0.0",

        # Validation
        "pydantic>=2.0.0,<3.0.0",

        # Observability
        "opentelemetry-api>=1.22.0,<2.0.0",
        "opentelemetry-sdk>=1.22.0,<2.0.0",
    ],

    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
        ],
        "observability": [
            "prometheus-client>=0.19.0",
            "opentelemetry-instrumentation-fastapi>=0.45b0",
        ],
    },

    entry_points={
        "console_scripts": [
            "ai-cost-controller=ai_cost_controller.bootstrap:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
