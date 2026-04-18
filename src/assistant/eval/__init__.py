"""Eval package — LLM-as-a-judge evaluation framework."""

from .dataset import EvalDataset, EvalQuestion
from .harness import HarnessEvent, HarnessResult, run_eval
from .judge import judge_response
from .metrics import ALL_METRICS, EvalScore, MetricName, MetricScore
from .report import print_detail, print_summary, save_report

__all__ = [
    "ALL_METRICS",
    "EvalDataset",
    "EvalQuestion",
    "EvalScore",
    "HarnessEvent",
    "HarnessResult",
    "MetricName",
    "MetricScore",
    "judge_response",
    "print_detail",
    "print_summary",
    "run_eval",
    "save_report",
]
