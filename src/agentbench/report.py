import csv
from pathlib import Path

from agentbench.metrics import CallLog, RunResult, Summary, summarize


def print_results_table(results: list[RunResult]) -> None:
    header = f"{'task':<20} {'approach':<8} {'ok':<4} {'score':>6} {'latency_s':>10} {'calls':>6} {'tokens':>8} {'cost_usd':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        tokens = r.input_tokens + r.output_tokens
        flag = "err" if r.error else ("yes" if r.success else "no")
        print(
            f"{r.task_id:<20} {r.approach:<8} {flag:<4} {r.score:>6.2f} "
            f"{r.latency_s:>10.2f} {r.num_calls:>6} {tokens:>8} {r.cost_usd:>9.4f}"
        )


def print_summary(summaries: list[Summary], judge_call_log: CallLog) -> None:
    print()
    header = f"{'approach':<8} {'n':>3} {'success%':>9} {'avg_score':>10} {'avg_latency_s':>14} {'avg_calls':>10} {'total_cost_usd':>15} {'total_tokens':>13}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.approach:<8} {s.n:>3} {s.success_rate * 100:>8.1f}% {s.avg_score:>10.2f} "
            f"{s.avg_latency_s:>14.2f} {s.avg_num_calls:>10.1f} {s.total_cost_usd:>15.4f} {s.total_tokens:>13}"
        )
    print(
        f"\n(judge/grading overhead, shared by both approaches: "
        f"{judge_call_log.num_calls} calls, ${judge_call_log.cost_usd:.4f})"
    )


def write_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


def write_markdown_summary(summaries: list[Summary], judge_call_log: CallLog, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| approach | n | success % | avg score | avg latency (s) | avg calls | total cost ($) | total tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.approach} | {s.n} | {s.success_rate * 100:.1f} | {s.avg_score:.2f} | "
            f"{s.avg_latency_s:.2f} | {s.avg_num_calls:.1f} | {s.total_cost_usd:.4f} | {s.total_tokens} |"
        )
    lines.append("")
    lines.append(
        f"Judge/grading overhead (shared by both approaches): "
        f"{judge_call_log.num_calls} calls, ${judge_call_log.cost_usd:.4f}"
    )
    path.write_text("\n".join(lines) + "\n")
