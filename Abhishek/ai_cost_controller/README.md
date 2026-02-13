# AI Cost Controller

Enterprise-grade AI orchestration and cost governance platform.

AI Cost Controller helps organizations manage, optimize, and govern large language model (LLM) usage across multiple providers with intelligent routing, budget enforcement, and cost tracking.

---

## 🚀 Features

- 🔁 Intelligent model orchestration
- 💰 Cost tracking & budget enforcement
- 📊 ROI scoring & logging
- 🧠 Automatic model switching
- 🔒 Enterprise-ready governance controls
- 🌐 Multi-provider support

---

## 📦 Installation

Install locally:

```bash
pip install .
```

Or build and install as a wheel:

```bash
python -m build
pip install dist/ai_cost_controller-*.whl
```

---

## 🧠 Quick Example

```python
from ai_cost_controller.execution.orchestrator import Orchestrator

orchestrator = Orchestrator()

response = orchestrator.run(
    prompt="Explain cost optimization strategies for AI systems."
)

print(response)
```

---

## 📂 Project Structure

```
ai_cost_controller/
│
├── execution/
│   └── orchestrator.py
│
├── policies/
├── providers/
├── logging/
└── utils/
```

---

## ⚙️ Use Cases

- Enterprise AI budget control
- Multi-model routing (e.g., Groq, Open-source LLMs)
- Token usage monitoring
- Cost-performance optimization
- Automated fallback systems

---

## 🏢 Ideal For

- AI startups
- SaaS platforms
- Enterprises using multiple LLM providers
- FinOps teams managing AI spend

---

## 📄 License

MIT License

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Support

For enterprise integration or customization, contact your internal AI platform team.
