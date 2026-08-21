"""Single-agent baseline: one Claude "role" does planning, execution, and
self-review in a loop, all under the same system prompt. This is the natural
baseline to compare a specialized multi-agent pipeline against, since both are
capped at the same max_iters budget.
"""

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agentbench.agents.llm import LLMClient

SYSTEM_PROMPT = (
    "You are a careful problem solver working alone. Read the task, produce "
    "your best answer, then critique your own work. End every response with "
    "exactly one line: 'STATUS: DONE' if you're confident the answer is "
    "correct and complete, or 'STATUS: REVISE' if you plan to improve it "
    "further."
)


class SingleAgentState(TypedDict):
    task_prompt: str
    max_iters: int
    iteration: int
    answer: str
    done: bool


def _strip_status(text: str) -> str:
    return re.sub(r"\n?STATUS:\s*(DONE|REVISE)\s*$", "", text.strip(), flags=re.IGNORECASE).strip()


def build_single_agent_graph(llm: LLMClient):
    def solve(state: SingleAgentState) -> dict:
        if state["iteration"] == 0:
            user_msg = state["task_prompt"]
        else:
            user_msg = (
                f"Original task:\n{state['task_prompt']}\n\n"
                f"Your previous answer:\n{state['answer']}\n\n"
                "Improve it further before finalizing."
            )

        response = llm.complete(system=SYSTEM_PROMPT, user=user_msg)
        iteration = state["iteration"] + 1
        is_done = bool(re.search(r"STATUS:\s*DONE", response.text, re.IGNORECASE))
        done = is_done or iteration >= state["max_iters"]

        return {
            "answer": _strip_status(response.text),
            "iteration": iteration,
            "done": done,
        }

    def route(state: SingleAgentState) -> str:
        return END if state["done"] else "solve"

    graph = StateGraph(SingleAgentState)
    graph.add_node("solve", solve)
    graph.set_entry_point("solve")
    graph.add_conditional_edges("solve", route, {"solve": "solve", END: END})
    return graph.compile()


def run_single_agent(llm: LLMClient, task_prompt: str, max_iters: int) -> str:
    graph = build_single_agent_graph(llm)
    final_state = graph.invoke(
        {
            "task_prompt": task_prompt,
            "max_iters": max_iters,
            "iteration": 0,
            "answer": "",
            "done": False,
        }
    )
    return final_state["answer"]
