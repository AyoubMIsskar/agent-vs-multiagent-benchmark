import time
from collections.abc import Iterable

from agentbench import config
from agentbench.agents.llm import ClaudeClient
from agentbench.agents.multi_agent import run_multi_agent
from agentbench.agents.single_agent import run_single_agent
from agentbench.metrics import CallLog, RunResult
from agentbench.tasks.base import Task

APPROACHES = {
    "single": run_single_agent,
    "multi": run_multi_agent,
}


def run_task(approach: str, task: Task, model: str, max_iters: int, judge_call_log: CallLog) -> RunResult:
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach '{approach}', expected one of {list(APPROACHES)}")

    call_log = CallLog()
    llm = ClaudeClient(call_log, model=model)
    judge = ClaudeClient(judge_call_log, model=model)

    start = time.perf_counter()
    output, error = "", None
    try:
        output = APPROACHES[approach](llm, task.build_prompt(), max_iters)
    except Exception as exc:  # a failed run is a data point, not a crash
        error = str(exc)
    latency_s = time.perf_counter() - start

    success, score = (False, 0.0) if error else task.evaluate(output, judge=judge)

    return RunResult(
        task_id=task.id,
        approach=approach,
        success=success,
        score=score,
        latency_s=latency_s,
        num_calls=call_log.num_calls,
        input_tokens=call_log.input_tokens,
        output_tokens=call_log.output_tokens,
        cost_usd=call_log.cost_usd,
        output=output,
        error=error,
    )


def run_benchmark(
    tasks: Iterable[Task],
    approaches: Iterable[str] = ("single", "multi"),
    model: str = config.DEFAULT_MODEL,
    max_iters: int = config.DEFAULT_MAX_ITERS,
    repeats: int = 1,
) -> tuple[list[RunResult], CallLog]:
    """Runs every (task x approach) pair `repeats` times.

    Returns the flat list of results plus a CallLog for the judge calls used
    during evaluation (tracked separately since it's evaluation overhead
    shared identically by both approaches, not agent execution cost).
    """
    judge_call_log = CallLog()
    results = []
    for _ in range(repeats):
        for task in tasks:
            for approach in approaches:
                results.append(run_task(approach, task, model, max_iters, judge_call_log))
    return results, judge_call_log
