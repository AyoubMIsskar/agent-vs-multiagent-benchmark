import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentbench.tasks.base import Task

if TYPE_CHECKING:
    from agentbench.agents.llm import LLMClient


@dataclass
class ArithmeticWordProblemTask(Task):
    """Deterministic: extract the final numeric answer and compare exactly."""

    id: str
    problem: str
    expected_answer: float

    def build_prompt(self) -> str:
        return (
            f"{self.problem}\n\n"
            "Think it through, then give the final numeric answer alone on the "
            "last line, formatted exactly as: ANSWER: <number>"
        )

    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        match = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", output)
        if not match:
            return False, 0.0
        got = float(match.group(1))
        success = abs(got - self.expected_answer) < 1e-6
        return success, 1.0 if success else 0.0


@dataclass
class ExactAnswerTask(Task):
    """Deterministic: compare a short text answer exactly (case-insensitive by
    default). Good for trivia, multi-hop QA, or any task with one correct
    short answer that isn't purely numeric."""

    id: str
    problem: str
    expected_answer: str
    case_sensitive: bool = False

    def build_prompt(self) -> str:
        return (
            f"{self.problem}\n\n"
            "Give the final answer alone on the last line, formatted exactly "
            "as: ANSWER: <answer>"
        )

    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        match = re.search(r"ANSWER:\s*(.+)", output)
        if not match:
            return False, 0.0
        got, expected = match.group(1).strip(), self.expected_answer.strip()
        if not self.case_sensitive:
            got, expected = got.lower(), expected.lower()
        success = got == expected
        return success, 1.0 if success else 0.0


@dataclass
class JsonExtractionTask(Task):
    """Deterministic: parse the JSON the agent produces and diff it against the
    expected fields. `instructions` defaults to a plain extraction prompt but
    can be overridden (e.g. to frame the same field-diff scoring around a
    puzzle to solve instead of a text to extract from)."""

    id: str
    source_text: str
    expected_fields: dict
    instructions: str | None = None

    def build_prompt(self) -> str:
        header = self.instructions or (
            f"Extract the following fields from the text below: "
            f"{', '.join(self.expected_fields.keys())}."
        )
        body = f"{header}\n\n{self.source_text}" if self.source_text else header
        return (
            f"{body}\n\n"
            "Respond with a single JSON object on the last line, and nothing "
            "else after it."
        )

    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if not json_match:
            return False, 0.0
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return False, 0.0

        matches = sum(
            1
            for key, expected in self.expected_fields.items()
            if str(parsed.get(key, "")).strip().lower() == str(expected).strip().lower()
        )
        score = matches / len(self.expected_fields)
        return score == 1.0, score


@dataclass
class ConstrainedWritingTask(Task):
    """Deterministic: check hard constraints on the generated text (length,
    required/forbidden words)."""

    id: str
    instructions: str
    max_words: int
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)

    def build_prompt(self) -> str:
        return (
            f"{self.instructions}\n\n"
            f"Constraints: at most {self.max_words} words; "
            f"must include the words: {', '.join(self.must_include) or 'none'}; "
            f"must NOT include the words: {', '.join(self.must_not_include) or 'none'}."
        )

    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        words = output.split()
        checks = [len(words) <= self.max_words]
        lowered = output.lower()
        checks += [w.lower() in lowered for w in self.must_include]
        checks += [w.lower() not in lowered for w in self.must_not_include]
        score = sum(checks) / len(checks) if checks else 0.0
        return all(checks), score


@dataclass
class FormattedListTask(Task):
    """Deterministic: check precise instruction-following on output shape —
    an exact bullet count, each bullet under a word limit."""

    id: str
    instructions: str
    required_bullet_count: int
    max_words_per_bullet: int

    def build_prompt(self) -> str:
        return (
            f"{self.instructions}\n\n"
            f"Format: exactly {self.required_bullet_count} bullet points, each "
            f"starting with '- ' and at most {self.max_words_per_bullet} words."
        )

    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        bullets = [line.strip()[2:] for line in output.splitlines() if line.strip().startswith("- ")]
        checks = [len(bullets) == self.required_bullet_count]
        checks += [len(b.split()) <= self.max_words_per_bullet for b in bullets]
        score = sum(checks) / len(checks) if checks else 0.0
        return all(checks), score


@dataclass
class OpenEndedJudgeTask(Task):
    """Open-ended: a second Claude call grades the answer 0-10 against a rubric.
    Use this shape for tasks that have no single correct answer."""

    id: str
    question: str
    rubric: str
    pass_threshold: float = 0.6

    def build_prompt(self) -> str:
        return self.question

    def evaluate(self, output: str, judge: "LLMClient | None" = None) -> tuple[bool, float]:
        if judge is None:
            raise ValueError(f"OpenEndedJudgeTask '{self.id}' requires a judge LLMClient")

        verdict = judge.complete(
            system=(
                "You are a strict grader. Score the ANSWER against the RUBRIC on "
                "a 0-10 scale. Respond with only the integer score, nothing else."
            ),
            user=f"QUESTION:\n{self.question}\n\nRUBRIC:\n{self.rubric}\n\nANSWER:\n{output}",
        )
        digits = re.search(r"\d+(?:\.\d+)?", verdict.text)
        raw_score = float(digits.group(0)) if digits else 0.0
        score = max(0.0, min(1.0, raw_score / 10))
        return score >= self.pass_threshold, score


def get_tasks() -> list[Task]:
    return [
        ArithmeticWordProblemTask(
            id="arithmetic_1",
            problem=(
                "A bakery bakes 240 loaves a day. It sells 60% of them in the "
                "morning, then 25% of what's left in the afternoon. The rest are "
                "donated. How many loaves are donated?"
            ),
            expected_answer=72,
        ),
        JsonExtractionTask(
            id="json_extraction_1",
            source_text=(
                "Invoice #4471, issued to Marine Dupont on 2026-03-04, total "
                "amount due: 812.50 EUR, payment terms: net 30."
            ),
            expected_fields={
                "invoice_number": "4471",
                "customer": "Marine Dupont",
                "total": "812.50",
                "currency": "EUR",
            },
        ),
        ConstrainedWritingTask(
            id="constrained_writing_1",
            instructions="Write a short product description for a mechanical keyboard.",
            max_words=60,
            must_include=["mechanical", "keys"],
            must_not_include=["cheap", "wireless"],
        ),
        OpenEndedJudgeTask(
            id="open_ended_1",
            question=(
                "Propose a rollout plan for migrating a mid-size company's "
                "internal wiki from Confluence to Notion, in under 200 words."
            ),
            rubric=(
                "A strong answer: sequences the migration in clear phases, "
                "addresses data export/import and permissions, calls out user "
                "training/communication, and flags at least one concrete risk."
            ),
        ),
        ExactAnswerTask(
            id="multihop_qa_1",
            problem=(
                "The Zenith Bridge was completed in 1978. The engineer who "
                "designed the Zenith Bridge, Maria Kovac, later designed the "
                "Lumen Tower, which was completed 16 years after the bridge.\n\n"
                "Question: In what year was the Lumen Tower completed?"
            ),
            expected_answer="1994",
        ),
        JsonExtractionTask(
            id="logic_puzzle_1",
            instructions=(
                "Solve the logic puzzle below. Alice, Ben, and Cara each drink "
                "a different beverage: coffee, tea, or water.\n"
                "1. Alice does not drink coffee.\n"
                "2. Ben drinks tea.\n"
                "3. Cara does not drink water.\n"
                "Determine what each person drinks."
            ),
            source_text="",
            expected_fields={"alice": "water", "ben": "tea", "cara": "coffee"},
        ),
        FormattedListTask(
            id="formatted_list_1",
            instructions=(
                "List the main risks of launching a product without a beta "
                "test phase."
            ),
            required_bullet_count=3,
            max_words_per_bullet=12,
        ),
        OpenEndedJudgeTask(
            id="tradeoff_1",
            question=(
                "A startup must choose between Postgres and MongoDB for a new "
                "product with evolving schema requirements and a small team. "
                "Recommend one, in under 150 words."
            ),
            rubric=(
                "A strong answer picks one option, justifies it against the "
                "stated constraints (evolving schema, small team), and "
                "acknowledges at least one concrete downside of the choice."
            ),
        ),
    ]


ALL_TASKS = get_tasks()
