"""LLM-as-a-judge evaluator — uses Claude to score assistant responses."""

from __future__ import annotations

import json
import re

from ..client import AssistantClient
from .metrics import (
    ALL_METRICS,
    METRIC_DESCRIPTIONS,
    EvalScore,
    MetricName,
    MetricScore,
)

_JUDGE_SYSTEM = """\
You are an expert evaluator of AI assistant responses. You will be given a question \
and a response from an AI assistant. Your job is to score the response on specific \
quality metrics using a 1–5 scale.

Scoring scale:
  1 — Very poor
  2 — Poor
  3 — Acceptable
  4 — Good
  5 — Excellent

You MUST respond with a valid JSON object in exactly this format:
{
  "scores": {
    "faithfulness":  {"score": <1-5>, "reasoning": "<one sentence>"},
    "relevance":     {"score": <1-5>, "reasoning": "<one sentence>"},
    "completeness":  {"score": <1-5>, "reasoning": "<one sentence>"},
    "conciseness":   {"score": <1-5>, "reasoning": "<one sentence>"}
  }
}

Rules:
- Be critical and calibrated. Reserve 5 for truly excellent responses.
- Each reasoning must be a single concise sentence explaining the score.
- If the response is empty or an error message, score everything 1.
- Respond with JSON only. No prose, no markdown fences.
"""


def _build_judge_prompt(
    question: str,
    response: str,
    reference_answer: str | None,
    context: str | None,
) -> str:
    parts = [f"Question:\n{question}", f"Assistant response:\n{response}"]
    if reference_answer:
        parts.append(f"Reference answer (for comparison):\n{reference_answer}")
    if context:
        parts.append(f"Source context the assistant had access to:\n{context[:1500]}")
    parts.append(
        "\nMetric descriptions:\n"
        + "\n".join(f"- {m.value}: {METRIC_DESCRIPTIONS[m]}" for m in ALL_METRICS)
    )
    return "\n\n".join(parts)


def judge_response(
    question_id: str,
    question: str,
    response: str,
    client: AssistantClient,
    reference_answer: str | None = None,
    context: str | None = None,
) -> EvalScore:
    """
    Ask Claude to score a single response on all metrics.

    Args:
        question_id:      Identifier for the eval question.
        question:         The question that was asked.
        response:         The assistant's response to evaluate.
        client:           AssistantClient to use for the judge call.
        reference_answer: Optional gold-standard answer for comparison.
        context:          Optional source context the assistant had access to.

    Returns:
        An EvalScore with per-metric scores and reasoning.
    """
    eval_score = EvalScore(
        question_id=question_id,
        question=question,
        response=response,
    )

    prompt = _build_judge_prompt(question, response, reference_answer, context)

    try:
        raw = client.complete(prompt=prompt, system=_JUDGE_SYSTEM)
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(raw)

        for metric in ALL_METRICS:
            entry = data["scores"].get(metric.value, {})
            score = int(entry.get("score", 1))
            score = max(1, min(5, score))   # clamp to [1, 5]
            reasoning = entry.get("reasoning", "No reasoning provided.")
            eval_score.scores.append(MetricScore(metric=metric, score=score, reasoning=reasoning))

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        eval_score.error = f"Judge parsing failed: {exc}"
        # Fill all metrics with score=1 so the run still produces a record
        for metric in ALL_METRICS:
            eval_score.scores.append(
                MetricScore(metric=metric, score=1, reasoning="Evaluation failed.")
            )

    return eval_score
