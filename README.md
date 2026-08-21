# agent-vs-multiagent-benchmark

Compare a **single-agent** baseline against a **multi-agent** pipeline (planner
→ executor → reviewer) built with [LangGraph](https://github.com/langchain-ai/langgraph)
and Claude, across a pluggable set of tasks.

## Why

Both approaches use the same underlying model and the same iteration budget
(`--max-iters`), so any difference in outcome comes from the *architecture*
(one role looping on itself vs. specialized roles collaborating), not from
extra model calls or a stronger model.

Metrics tracked per run:

- **Quality**: success (pass/fail) and a continuous score in [0, 1], defined
  per task (exact match, structural checks, or an LLM-as-judge rubric).
- **Cost**: input/output tokens and USD cost (see `src/agentbench/config.py`
  for the pricing table).
- **Latency**: wall-clock seconds for the full run (all LLM calls in the
  graph, sequential).
- **Calls**: number of LLM calls used to reach a final answer.

## Architecture

```
src/agentbench/
  tasks/          Task adapter interface + example tasks (plug your own here)
  agents/
    llm.py        LLMClient abstraction (ClaudeClient wraps langchain-anthropic)
    single_agent.py  One role, self-loop until DONE or max_iters (LangGraph)
    multi_agent.py   plan -> execute -> review loop until APPROVE or max_iters
  runner.py       Runs (task x approach) combinations, collects RunResults
  report.py       Console table, CSV, markdown summary
  metrics.py      RunResult / Summary dataclasses
main.py           CLI entrypoint
tests/            Graph control-flow tests using a scripted fake LLM (no API calls)
```

### Adding your own task

Implement `Task` (`src/agentbench/tasks/base.py`):

```python
class MyTask(Task):
    id = "my_task"

    def build_prompt(self) -> str:
        return "..."  # what both agents receive

    def evaluate(self, output: str, judge: LLMClient | None = None) -> tuple[bool, float]:
        ...  # return (success, score in [0, 1])
```

Deterministic tasks (exact match, JSON diff, constraint checks) ignore
`judge`. Open-ended tasks can call `judge.complete(...)` to grade with an
LLM-as-judge rubric — see `OpenEndedJudgeTask` in `tasks/examples.py`.

Then register it in `get_tasks()` in the same file (or build your own list
and pass it to `run_benchmark` directly).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Run

```bash
python main.py                                   # all tasks, both approaches
python main.py --tasks arithmetic_1 json_extraction_1
python main.py --approaches multi --max-iters 5 --repeats 3
```

Results print to the console and are written to `results/results.csv` and
`results/summary.md`.

## Tests

```bash
pytest
```

Tests exercise the LangGraph control flow (loop-until-done, loop-until-approve,
max-iters cutoff) with a scripted fake `LLMClient`, so they run without
network access or an API key.
