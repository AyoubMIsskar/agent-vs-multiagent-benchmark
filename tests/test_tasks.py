from agentbench.tasks.examples import (
    ArithmeticWordProblemTask,
    ConstrainedWritingTask,
    ExactAnswerTask,
    FormattedListTask,
    JsonExtractionTask,
    OpenEndedJudgeTask,
    get_tasks,
)
from tests.fakes import ScriptedLLMClient


def test_get_tasks_has_unique_ids():
    tasks = get_tasks()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    assert len(tasks) >= 8


def test_arithmetic_task_exact_and_wrong():
    task = ArithmeticWordProblemTask(id="t", problem="p", expected_answer=72)
    assert task.evaluate("blah\nANSWER: 72") == (True, 1.0)
    assert task.evaluate("blah\nANSWER: 71") == (False, 0.0)
    assert task.evaluate("no answer line here") == (False, 0.0)


def test_exact_answer_task_case_insensitive_by_default():
    task = ExactAnswerTask(id="t", problem="p", expected_answer="1994")
    assert task.evaluate("some reasoning\nANSWER: 1994")[0] is True
    assert task.evaluate("some reasoning\nANSWER: 1995")[0] is False


def test_exact_answer_task_case_sensitive():
    task = ExactAnswerTask(id="t", problem="p", expected_answer="Paris", case_sensitive=True)
    assert task.evaluate("ANSWER: paris")[0] is False
    assert task.evaluate("ANSWER: Paris")[0] is True


def test_json_extraction_partial_credit():
    task = JsonExtractionTask(
        id="t",
        source_text="irrelevant",
        expected_fields={"a": "1", "b": "2"},
    )
    success, score = task.evaluate('noise before {"a": "1", "b": "wrong"} noise after')
    assert success is False
    assert score == 0.5


def test_json_extraction_full_match_with_custom_instructions():
    task = JsonExtractionTask(
        id="t",
        source_text="",
        instructions="Solve the puzzle.",
        expected_fields={"alice": "water", "ben": "tea"},
    )
    success, score = task.evaluate('{"alice": "Water", "ben": "Tea"}')
    assert success is True
    assert score == 1.0
    assert "Solve the puzzle." in task.build_prompt()


def test_json_extraction_invalid_json():
    task = JsonExtractionTask(id="t", source_text="x", expected_fields={"a": "1"})
    assert task.evaluate("not json at all") == (False, 0.0)


def test_constrained_writing_task():
    task = ConstrainedWritingTask(
        id="t",
        instructions="write it",
        max_words=5,
        must_include=["mechanical"],
        must_not_include=["cheap"],
    )
    assert task.evaluate("mechanical keyboards are great")[0] is True
    assert task.evaluate("this cheap mechanical keyboard is bad")[0] is False
    assert task.evaluate("mechanical one two three four five six")[0] is False


def test_formatted_list_task_exact_shape():
    task = FormattedListTask(id="t", instructions="list risks", required_bullet_count=3, max_words_per_bullet=4)
    good = "- risk one here now\n- risk two here now\n- risk three here now"
    success, score = task.evaluate(good)
    assert success is True
    assert score == 1.0


def test_formatted_list_task_wrong_bullet_count():
    task = FormattedListTask(id="t", instructions="list risks", required_bullet_count=3, max_words_per_bullet=10)
    success, score = task.evaluate("- only one bullet")
    assert success is False
    assert score < 1.0


def test_open_ended_judge_task_uses_judge_and_thresholds():
    task = OpenEndedJudgeTask(id="t", question="q", rubric="r", pass_threshold=0.6)
    judge = ScriptedLLMClient(["8"])
    success, score = task.evaluate("some answer", judge=judge)
    assert success is True
    assert score == 0.8
    assert judge.call_log.num_calls == 1


def test_open_ended_judge_task_below_threshold_fails():
    task = OpenEndedJudgeTask(id="t", question="q", rubric="r", pass_threshold=0.6)
    judge = ScriptedLLMClient(["3"])
    success, score = task.evaluate("weak answer", judge=judge)
    assert success is False
    assert score == 0.3


def test_open_ended_judge_task_requires_judge():
    task = OpenEndedJudgeTask(id="t", question="q", rubric="r")
    try:
        task.evaluate("answer", judge=None)
        assert False, "expected ValueError"
    except ValueError:
        pass
