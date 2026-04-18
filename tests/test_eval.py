"""Unit tests for the eval framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistant.eval.dataset import EvalDataset, EvalQuestion
from assistant.eval.harness import HarnessResult, run_eval
from assistant.eval.judge import judge_response
from assistant.eval.metrics import (
    ALL_METRICS,
    EvalScore,
    MetricName,
    MetricScore,
)
from assistant.eval.report import print_summary, save_report


# ---------------------------------------------------------------------------
# EvalQuestion / EvalDataset
# ---------------------------------------------------------------------------

def test_question_round_trip():
    q = EvalQuestion(id="q1", question="What is AI?", tags=["basics"])
    assert EvalQuestion.from_dict(q.to_dict()).id == "q1"
    assert EvalQuestion.from_dict(q.to_dict()).tags == ["basics"]


def test_dataset_len():
    ds = EvalDataset(name="test", questions=[
        EvalQuestion(id="q1", question="Q1"),
        EvalQuestion(id="q2", question="Q2"),
    ])
    assert len(ds) == 2


def test_dataset_filter_by_tag():
    ds = EvalDataset(name="test", questions=[
        EvalQuestion(id="q1", question="Q1", tags=["rag"]),
        EvalQuestion(id="q2", question="Q2", tags=["agents"]),
    ])
    filtered = ds.filter_by_tag("rag")
    assert len(filtered) == 1
    assert filtered.questions[0].id == "q1"


def test_dataset_save_and_load(tmp_path):
    ds = EvalDataset(name="save_test", questions=[
        EvalQuestion(id="q1", question="Test question?", reference_answer="Answer."),
    ])
    path = tmp_path / "ds.json"
    ds.save(path)
    loaded = EvalDataset.load(path)
    assert loaded.name == "save_test"
    assert loaded.questions[0].question == "Test question?"
    assert loaded.questions[0].reference_answer == "Answer."


def test_dataset_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        EvalDataset.load(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# EvalScore / MetricScore
# ---------------------------------------------------------------------------

def test_eval_score_mean():
    es = EvalScore(question_id="q1", question="Q", response="A")
    for i, metric in enumerate(ALL_METRICS, start=2):
        es.scores.append(MetricScore(metric=metric, score=i, reasoning="ok"))
    assert es.mean == pytest.approx(sum(range(2, 2 + len(ALL_METRICS))) / len(ALL_METRICS))


def test_eval_score_mean_empty():
    es = EvalScore(question_id="q1", question="Q", response="A")
    assert es.mean == 0.0


def test_eval_score_score_for():
    es = EvalScore(question_id="q1", question="Q", response="A")
    es.scores.append(MetricScore(metric=MetricName.FAITHFULNESS, score=4, reasoning="good"))
    assert es.score_for(MetricName.FAITHFULNESS) == 4
    assert es.score_for(MetricName.RELEVANCE) is None


def test_eval_score_to_dict_keys():
    es = EvalScore(question_id="q1", question="Q", response="A")
    es.scores.append(MetricScore(metric=MetricName.FAITHFULNESS, score=3, reasoning="ok"))
    d = es.to_dict()
    assert "question_id" in d
    assert "mean_score" in d
    assert "faithfulness" in d["scores"]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _mock_judge_client(json_response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = json_response
    return client


def test_judge_response_parses_scores():
    response_json = json.dumps({
        "scores": {
            "faithfulness": {"score": 4, "reasoning": "Mostly accurate."},
            "relevance": {"score": 5, "reasoning": "Directly on topic."},
            "completeness": {"score": 3, "reasoning": "Missing some detail."},
            "conciseness": {"score": 4, "reasoning": "Appropriately brief."},
        }
    })
    client = _mock_judge_client(response_json)
    result = judge_response("q1", "What is AI?", "AI is...", client)

    assert result.score_for(MetricName.FAITHFULNESS) == 4
    assert result.score_for(MetricName.RELEVANCE) == 5
    assert result.error is None


def test_judge_response_clamps_scores():
    response_json = json.dumps({
        "scores": {
            "faithfulness": {"score": 99, "reasoning": "Great."},
            "relevance": {"score": -1, "reasoning": "Bad."},
            "completeness": {"score": 3, "reasoning": "OK."},
            "conciseness": {"score": 3, "reasoning": "OK."},
        }
    })
    client = _mock_judge_client(response_json)
    result = judge_response("q1", "Q", "A", client)

    assert result.score_for(MetricName.FAITHFULNESS) == 5
    assert result.score_for(MetricName.RELEVANCE) == 1


def test_judge_response_handles_bad_json():
    client = _mock_judge_client("not json at all")
    result = judge_response("q1", "Q", "A", client)

    assert result.error is not None
    # All metrics should still be present (defaulted to 1)
    assert len(result.scores) == len(ALL_METRICS)


def test_judge_response_strips_markdown_fences():
    response_json = "```json\n" + json.dumps({
        "scores": {m.value: {"score": 3, "reasoning": "ok"} for m in ALL_METRICS}
    }) + "\n```"
    client = _mock_judge_client(response_json)
    result = judge_response("q1", "Q", "A", client)
    assert result.error is None


# ---------------------------------------------------------------------------
# HarnessResult
# ---------------------------------------------------------------------------

def _make_result(scores_per_q: list[list[int]]) -> HarnessResult:
    result = HarnessResult(dataset_name="test")
    for i, scores in enumerate(scores_per_q):
        es = EvalScore(question_id=f"q{i}", question="Q", response="A")
        for metric, score in zip(ALL_METRICS, scores):
            es.scores.append(MetricScore(metric=metric, score=score, reasoning="ok"))
        result.eval_scores.append(es)
    return result


def test_harness_result_mean_overall():
    result = _make_result([[4, 4, 4, 4], [2, 2, 2, 2]])
    assert result.mean_overall == pytest.approx(3.0)


def test_harness_result_mean_for_metric():
    result = _make_result([[5, 3, 3, 3], [5, 3, 3, 3]])
    assert result.mean_for_metric(MetricName.FAITHFULNESS) == pytest.approx(5.0)


def test_harness_result_summary_keys():
    result = _make_result([[3, 3, 3, 3]])
    summary = result.summary()
    assert "mean_overall" in summary
    assert "metrics" in summary
    assert "faithfulness" in summary["metrics"]


def test_harness_result_empty():
    result = HarnessResult(dataset_name="empty")
    assert result.mean_overall == 0.0


# ---------------------------------------------------------------------------
# run_eval harness
# ---------------------------------------------------------------------------

def test_run_eval_yields_events_and_returns_result():
    judge_json = json.dumps({
        "scores": {m.value: {"score": 4, "reasoning": "Good."} for m in ALL_METRICS}
    })

    assistant_client = MagicMock()
    judge_client = MagicMock()
    judge_client.complete.return_value = judge_json

    # run_with_tools yields one chunk per call
    assistant_client.run_with_tools.return_value = iter(["The answer is here."])

    registry = MagicMock()

    dataset = EvalDataset(name="mini", questions=[
        EvalQuestion(id="q1", question="What is RAG?"),
        EvalQuestion(id="q2", question="What is fine-tuning?"),
    ])

    gen = run_eval(dataset, assistant_client, judge_client, registry)
    events = []
    result = None

    try:
        while True:
            events.append(next(gen))
    except StopIteration as e:
        result = e.value

    assert len(events) == 2
    assert result is not None
    assert len(result.eval_scores) == 2
    assert result.mean_overall > 0


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_writes_json(tmp_path):
    result = _make_result([[4, 4, 4, 4]])
    path = tmp_path / "report.json"
    save_report(result, path)
    data = json.loads(path.read_text())
    assert "mean_overall" in data
    assert "results" in data
