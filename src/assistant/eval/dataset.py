"""Eval dataset — load, validate, and save sets of evaluation questions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuestion:
    id: str
    question: str
    reference_answer: str | None = None   # Gold-standard answer for the judge
    context_hint: str | None = None       # Optional hint about which docs are relevant
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "context_hint": self.context_hint,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalQuestion":
        return cls(
            id=data["id"],
            question=data["question"],
            reference_answer=data.get("reference_answer"),
            context_hint=data.get("context_hint"),
            tags=data.get("tags", []),
        )


@dataclass
class EvalDataset:
    name: str
    questions: list[EvalQuestion] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    def filter_by_tag(self, tag: str) -> "EvalDataset":
        return EvalDataset(
            name=f"{self.name}[{tag}]",
            questions=[q for q in self.questions if tag in q.tags],
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalDataset":
        return cls(
            name=data.get("name", "unnamed"),
            questions=[EvalQuestion.from_dict(q) for q in data.get("questions", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        data = json.loads(path.read_text())
        return cls.from_dict(data)
