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


def _extract_text(content) -> str:
    """`response.content` is a plain string for most models, but a list of
    content blocks (text/thinking/etc.) for models with extended thinking.
    Concatenate just the text blocks in that case."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class ClaudeClient(LLMClient):
    """Real Claude-backed client (via langchain-anthropic)."""

    def __init__(self, call_log: CallLog, model: str = config.DEFAULT_MODEL):
        super().__init__(call_log)
        from langchain_anthropic import ChatAnthropic

        self.model = model
        # Newer Claude models (e.g. claude-sonnet-5) reject `temperature`
        # entirely, so it's left at the API default rather than passed here.
        self._llm = ChatAnthropic(model=model)

    def complete(self, system: str, user: str) -> LLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = config.cost_usd(self.model, input_tokens, output_tokens)
        self.call_log.record(input_tokens, output_tokens, cost)

        return LLMResponse(text=_extract_text(response.content))
