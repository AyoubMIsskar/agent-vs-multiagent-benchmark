from agentbench.agents.llm import LLMClient, LLMResponse
from agentbench.metrics import CallLog


class ScriptedLLMClient(LLMClient):
    """Returns canned responses in order, so graph/task logic can be tested
    without hitting the real Anthropic API."""

    def __init__(self, responses: list[str]):
        super().__init__(CallLog())
        self._responses = list(responses)

    def complete(self, system: str, user: str) -> LLMResponse:
        text = self._responses.pop(0)
        self.call_log.record(input_tokens=10, output_tokens=10, cost_usd=0.001)
        return LLMResponse(text=text)
