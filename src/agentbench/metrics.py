from dataclasses import dataclass, field


@dataclass
class CallLog:
    """Running totals for one graph execution (one task x one approach)."""

    num_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def record(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.num_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd


@dataclass
class RunResult:
    task_id: str
    approach: str
    success: bool
    score: float
    latency_s: float
    num_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    output: str
    error: str | None = None


@dataclass
class Summary:
    approach: str
    n: int
    success_rate: float
    avg_score: float
    avg_latency_s: float
    avg_num_calls: float
    total_cost_usd: float
    total_tokens: int


def summarize(results: list[RunResult]) -> list[Summary]:
    by_approach: dict[str, list[RunResult]] = {}
    for r in results:
        by_approach.setdefault(r.approach, []).append(r)

    summaries = []
    for approach, rs in by_approach.items():
        n = len(rs)
        summaries.append(
            Summary(
                approach=approach,
                n=n,
                success_rate=sum(r.success for r in rs) / n,
                avg_score=sum(r.score for r in rs) / n,
                avg_latency_s=sum(r.latency_s for r in rs) / n,
                avg_num_calls=sum(r.num_calls for r in rs) / n,
                total_cost_usd=sum(r.cost_usd for r in rs),
                total_tokens=sum(r.input_tokens + r.output_tokens for r in rs),
            )
        )
    return sorted(summaries, key=lambda s: s.approach)
