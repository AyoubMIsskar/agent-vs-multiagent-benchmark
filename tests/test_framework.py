from agentbench.agents.multi_agent import run_multi_agent
from agentbench.agents.single_agent import run_single_agent
from agentbench.metrics import RunResult, summarize
from tests.fakes import ScriptedLLMClient


def test_single_agent_stops_on_done():
    llm = ScriptedLLMClient(["42\nSTATUS: DONE"])
    answer = run_single_agent(llm, "what is 6*7?", max_iters=3)
    assert answer == "42"
    assert llm.call_log.num_calls == 1


def test_single_agent_stops_at_max_iters_if_never_done():
    llm = ScriptedLLMClient(
        [
            "draft 1\nSTATUS: REVISE",
            "draft 2\nSTATUS: REVISE",
            "draft 3\nSTATUS: REVISE",
        ]
    )
    answer = run_single_agent(llm, "write something", max_iters=3)
    assert answer == "draft 3"
    assert llm.call_log.num_calls == 3


def test_multi_agent_stops_on_approve():
    llm = ScriptedLLMClient(
        [
            "1. step one\n2. step two",  # plan
            "final answer",  # execute
            "looks good\nSTATUS: APPROVE",  # review
        ]
    )
    answer = run_multi_agent(llm, "do the thing", max_iters=3)
    assert answer == "final answer"
    assert llm.call_log.num_calls == 3


def test_multi_agent_loops_until_approve():
    llm = ScriptedLLMClient(
        [
            "plan",
            "draft 1",
            "needs work\nSTATUS: REVISE",
            "draft 2",
            "looks good\nSTATUS: APPROVE",
        ]
    )
    answer = run_multi_agent(llm, "do the thing", max_iters=3)
    assert answer == "draft 2"
    assert llm.call_log.num_calls == 5


def test_multi_agent_respects_max_iters_without_approval():
    llm = ScriptedLLMClient(
        [
            "plan",
            "draft 1",
            "needs work\nSTATUS: REVISE",
            "draft 2",
        ]
    )
    answer = run_multi_agent(llm, "do the thing", max_iters=2)
    assert answer == "draft 2"
    assert llm.call_log.num_calls == 4


def test_summarize_aggregates_by_approach():
    results = [
        RunResult("t1", "single", True, 1.0, 1.0, 1, 10, 10, 0.01, "ok"),
        RunResult("t2", "single", False, 0.0, 2.0, 3, 30, 30, 0.03, "no"),
        RunResult("t1", "multi", True, 1.0, 3.0, 5, 50, 50, 0.05, "ok"),
    ]
    summaries = {s.approach: s for s in summarize(results)}

    assert summaries["single"].n == 2
    assert summaries["single"].success_rate == 0.5
    assert summaries["multi"].n == 1
    assert summaries["multi"].success_rate == 1.0
    assert summaries["multi"].total_cost_usd == 0.05
