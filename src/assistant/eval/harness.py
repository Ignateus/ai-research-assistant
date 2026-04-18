"""Eval harness — runs the assistant against a dataset and collects scores."""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass, field
from statistics import mean, stdev

from ..client import AssistantClient, ConversationSession
from ..tools.registry import ToolRegistry
from .dataset import EvalDataset, EvalQuestion
from .judge import judge_response
from .metrics import ALL_METRICS, EvalScore, MetricName

_ASSISTANT_SYSTEM = """\
You are an expert research assistant. Answer questions clearly, accurately, \
and concisely. Use tools when they would improve the quality of your answer.\
"""


@dataclass
class HarnessEvent:
    """Emitted by the harness generator so callers can show progress."""
    question: EvalQuestion
    response: str
    eval_score: EvalScore
    elapsed_seconds: float


@dataclass
class HarnessResult:
    """Aggregated results for an entire eval run."""

    dataset_name: str
    eval_scores: list[EvalScore] = field(default_factory=list)

    @property
    def mean_overall(self) -> float:
        if not self.eval_scores:
            return 0.0
        return mean(s.mean for s in self.eval_scores)

    def mean_for_metric(self, metric: MetricName) -> float:
        scores = [s.score_for(metric) for s in self.eval_scores]
        valid = [s for s in scores if s is not None]
        return mean(valid) if valid else 0.0

    def std_for_metric(self, metric: MetricName) -> float:
        scores = [s.score_for(metric) for s in self.eval_scores]
        valid = [s for s in scores if s is not None]
        return stdev(valid) if len(valid) >= 2 else 0.0

    def summary(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "n_questions": len(self.eval_scores),
            "mean_overall": round(self.mean_overall, 2),
            "metrics": {
                m.value: {
                    "mean": round(self.mean_for_metric(m), 2),
                    "std": round(self.std_for_metric(m), 2),
                }
                for m in ALL_METRICS
            },
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "results": [s.to_dict() for s in self.eval_scores],
        }


def run_eval(
    dataset: EvalDataset,
    assistant_client: AssistantClient,
    judge_client: AssistantClient,
    registry: ToolRegistry,
) -> Generator[HarnessEvent, None, HarnessResult]:
    """
    Run the assistant against every question in the dataset, then judge each response.

    Yields a HarnessEvent per question so callers can display live progress.
    Returns the final HarnessResult.

    The assistant and judge can share the same client — they are separated
    here so in production you could use a cheaper model for one of them.

    Args:
        dataset:          EvalDataset to evaluate against.
        assistant_client: Client used to generate answers.
        judge_client:     Client used to score answers (LLM-as-a-judge).
        registry:         ToolRegistry available to the assistant.
    """
    result = HarnessResult(dataset_name=dataset.name)

    for question in dataset.questions:
        t0 = time.monotonic()

        # ── Run the assistant ─────────────────────────────────────────────
        session = ConversationSession(system_prompt=_ASSISTANT_SYSTEM)
        session.add_user(question.question)

        response_parts: list[str] = []
        try:
            for chunk in assistant_client.run_with_tools(session, registry):
                response_parts.append(chunk)
            response = "".join(response_parts).strip()
        except Exception as exc:  # noqa: BLE001
            response = f"[Error generating response: {exc}]"

        # ── Judge the response ────────────────────────────────────────────
        eval_score = judge_response(
            question_id=question.id,
            question=question.question,
            response=response,
            client=judge_client,
            reference_answer=question.reference_answer,
        )

        result.eval_scores.append(eval_score)
        elapsed = time.monotonic() - t0

        yield HarnessEvent(
            question=question,
            response=response,
            eval_score=eval_score,
            elapsed_seconds=elapsed,
        )

    return result
