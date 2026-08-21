"""Multi-agent baseline: three specialized Claude roles (planner, executor,
reviewer) collaborate through a shared state. `max_iters` bounds the number of
executor calls, so it's directly comparable to the single agent's solve-call
budget in single_agent.py; the planner/reviewer calls are the extra overhead
that specialization costs.
"""

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agentbench.agents.llm import LLMClient

PLANNER_SYSTEM = (
    "You are a planner. Break the task below into a short, numbered plan (3-5 "
    "steps) for another agent to execute. Do not solve the task yourself."
)

EXECUTOR_SYSTEM = (
    "You are an executor. Follow the plan to produce a final answer to the "
    "task. If reviewer feedback is provided, address it directly."
)

REVIEWER_SYSTEM = (
    "You are a reviewer. Check the executor's answer against the original "
    "task. End your response with exactly one line: 'STATUS: APPROVE' if the "
    "answer fully satisfies the task, or 'STATUS: REVISE' otherwise. If "
    "REVISE, first give concise, actionable feedback."
)


class MultiAgentState(TypedDict):
    task_prompt: str
    max_iters: int
    iteration: int
    plan: str
    answer: str
    feedback: str
    done: bool


def _strip_status(text: str) -> str:
    return re.sub(r"\n?STATUS:\s*(APPROVE|REVISE)\s*$", "", text.strip(), flags=re.IGNORECASE).strip()


def build_multi_agent_graph(llm: LLMClient):
    def plan(state: MultiAgentState) -> dict:
        response = llm.complete(system=PLANNER_SYSTEM, user=state["task_prompt"])
        return {"plan": response.text}

    def execute(state: MultiAgentState) -> dict:
        user_msg = f"Task:\n{state['task_prompt']}\n\nPlan:\n{state['plan']}"
        if state["feedback"]:
            user_msg += f"\n\nReviewer feedback to address:\n{state['feedback']}"
        response = llm.complete(system=EXECUTOR_SYSTEM, user=user_msg)
        return {"answer": response.text, "iteration": state["iteration"] + 1}

    def review(state: MultiAgentState) -> dict:
        if state["iteration"] >= state["max_iters"]:
            return {"done": True}

        user_msg = f"Task:\n{state['task_prompt']}\n\nAnswer:\n{state['answer']}"
        response = llm.complete(system=REVIEWER_SYSTEM, user=user_msg)
        approved = bool(re.search(r"STATUS:\s*APPROVE", response.text, re.IGNORECASE))
        return {"feedback": _strip_status(response.text), "done": approved}

    def route(state: MultiAgentState) -> str:
        return END if state["done"] else "execute"

    graph = StateGraph(MultiAgentState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("review", review)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", route, {"execute": "execute", END: END})
    return graph.compile()


def run_multi_agent(llm: LLMClient, task_prompt: str, max_iters: int) -> str:
    graph = build_multi_agent_graph(llm)
    final_state = graph.invoke(
        {
            "task_prompt": task_prompt,
            "max_iters": max_iters,
            "iteration": 0,
            "plan": "",
            "answer": "",
            "feedback": "",
            "done": False,
        }
    )
    return final_state["answer"]
