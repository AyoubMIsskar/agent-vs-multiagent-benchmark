from abc import ABC, abstractmethod
from dataclasses import dataclass

from agentbench import config
from agentbench.metrics import CallLog


@dataclass
class LLMResponse:
    text: str


class LLMClient(ABC):
    """Chat-completion client that records usage into a CallLog for metrics."""

    def __init__(self, call_log: CallLog):
        self.call_log = call_log

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse: ...


class ClaudeClient(LLMClient):
    """Real Claude-backed client (via langchain-anthropic)."""

    def __init__(self, call_log: CallLog, model: str = config.DEFAULT_MODEL, temperature: float = config.DEFAULT_TEMPERATURE):
        super().__init__(call_log)
        from langchain_anthropic import ChatAnthropic

        self.model = model
        self._llm = ChatAnthropic(model=model, temperature=temperature)

    def complete(self, system: str, user: str) -> LLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = config.cost_usd(self.model, input_tokens, output_tokens)
        self.call_log.record(input_tokens, output_tokens, cost)

        return LLMResponse(text=response.content)
