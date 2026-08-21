from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentbench.agents.llm import LLMClient


class Task(ABC):
    """Adapter interface: implement one class per task family to plug it into the
    benchmark. `build_prompt` is what both approaches receive as the problem
    statement; `evaluate` scores whatever final answer they produce.
    """

    id: str

    @abstractmethod
    def build_prompt(self) -> str:
        """The problem statement handed to the agent(s)."""

    @abstractmethod
    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        """Return (success, score in [0, 1]) for a final answer.

        `judge` is an optional LLMClient the task may use to grade open-ended
        output (e.g. an LLM-as-judge rubric). Deterministic tasks can ignore it.
        """
