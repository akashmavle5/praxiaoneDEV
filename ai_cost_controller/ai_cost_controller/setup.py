from setuptools import setup, find_packages

setup(
    name="ai_cost_controller",
    version="0.1.0",
    description="Enterprise AI orchestration and cost governance platform",
    author="Your Name",
    author_email="your@email.com",
    packages=find_packages(),
    install_requires=[
        "groq>=0.11.0,<1.0.0",
        "requests>=2.31.0,<3.0.0",
        "sqlalchemy>=2.0.0,<3.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
