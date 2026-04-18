"""Metric definitions for evaluating assistant responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MetricName(str, Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"


# Human-readable descriptions sent to the judge
METRIC_DESCRIPTIONS: dict[MetricName, str] = {
    MetricName.FAITHFULNESS: (
        "Does the response contain only factually accurate information? "
        "Penalise hallucinations, fabricated citations, or contradictions with the context."
    ),
    MetricName.RELEVANCE: (
        "Does the response directly address the question asked? "
        "Penalise off-topic content or answers to a different question."
    ),
    MetricName.COMPLETENESS: (
        "Does the response cover all aspects of the question? "
        "Penalise partial answers that leave key parts unaddressed."
    ),
    MetricName.CONCISENESS: (
        "Is the response appropriately concise — no unnecessary padding, repetition, "
        "or filler? Penalise verbose responses that dilute the core answer."
    ),
}

ALL_METRICS = list(MetricName)


@dataclass
class MetricScore:
    metric: MetricName
    score: int          # 1–5
    reasoning: str      # Judge's one-sentence explanation

    def __str__(self) -> str:
        return f"{self.metric.value}: {self.score}/5 — {self.reasoning}"


@dataclass
class EvalScore:
    """All metric scores for a single question/response pair."""

    question_id: str
    question: str
    response: str
    scores: list[MetricScore] = field(default_factory=list)
    error: str | None = None

    @property
    def mean(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def score_for(self, metric: MetricName) -> int | None:
        for s in self.scores:
            if s.metric == metric:
                return s.score
        return None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "response": self.response[:500],
            "mean_score": round(self.mean, 2),
            "scores": {s.metric.value: {"score": s.score, "reasoning": s.reasoning} for s in self.scores},
            "error": self.error,
        }
