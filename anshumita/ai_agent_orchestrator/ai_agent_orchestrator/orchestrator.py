class AgentOrchestrator:
    def __init__(self, budget=1.0, risk=0.2):
        self.budget = budget
        self.risk = risk

    def plan(self):
        model = "gpt-3.5" if self.budget < 0.5 else "gpt-4"
        verifier = True if self.risk > 0.7 else False
        return {"model": model, "verifier": verifier}

    def run(self):
        config = self.plan()
        print("Running Agent Orchestrator with:", config)
