"""Typed questions. Each renders to prompt text plus answer-token variants,
and interprets a probability vector into a typed, schema-valid value."""

from __future__ import annotations

import string
from dataclasses import dataclass

# Single-letter keys tokenize to one token in every mainstream tokenizer.
KEYS = list(string.ascii_uppercase) + list(string.ascii_lowercase)


@dataclass
class Question:
    key: str
    question: str

    def render(self) -> tuple[str, list[list[str]]]:
        raise NotImplementedError

    def interpret(self, probs: list[float]) -> dict:
        raise NotImplementedError


@dataclass
class Bool(Question):
    def render(self):
        text = f"Question: {self.question}\nAnswer yes or no."
        return text, [["yes", "Yes", "YES"], ["no", "No", "NO"]]

    def interpret(self, probs):
        return {"type": "bool", "p_true": round(probs[0], 4), "value": probs[0] >= 0.5}


@dataclass
class Choice(Question):
    options: list[str] | None = None

    def __post_init__(self):
        if not self.options:
            raise ValueError(f"{self.key}: Choice needs at least 1 option")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"{self.key}: duplicate options")
        if len(self.options) > len(KEYS):
            raise ValueError(f"{self.key}: at most {len(KEYS)} options; use shortlist() for more")

    def render(self):
        lines = [f"Question: {self.question}", "Options:"]
        for k, opt in zip(KEYS, self.options):
            lines.append(f"{k}) {opt}")
        lines.append("Answer with the letter of the best option only.")
        return "\n".join(lines), [[k] for k in KEYS[: len(self.options)]]

    def interpret(self, probs):
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "choice": self.options[best],
            "confidence": round(probs[best], 4),
            "probabilities": {o: round(p, 4) for o, p in zip(self.options, probs)},
        }


@dataclass
class Score(Question):
    legend: list[str] | None = None  # legend[i] describes level i

    def __post_init__(self):
        if not self.legend or not 2 <= len(self.legend) <= 10:
            raise ValueError(f"{self.key}: Score legend needs 2..10 levels")

    def render(self):
        lines = [f"Question: {self.question}", "Scale:"]
        for i, desc in enumerate(self.legend):
            lines.append(f"{i} = {desc}")
        lines.append("Answer with the single digit only.")
        return "\n".join(lines), [[str(i)] for i in range(len(self.legend))]

    def interpret(self, probs):
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "score",
            "score": round(sum(i * p for i, p in enumerate(probs)), 3),
            "level": best,
            "confidence": round(probs[best], 4),
            "legend": dict(enumerate(self.legend)),
        }


def to_spec(q: Question) -> dict:
    """JSON-serializable form of a question (for datasets and remote jobs)."""
    d = {"key": q.key, "type": type(q).__name__.lower(), "question": q.question}
    if isinstance(q, Choice):
        d["options"] = list(q.options)
    elif isinstance(q, Score):
        d["legend"] = list(q.legend)
    return d


def from_spec(d: dict) -> Question:
    kind = d["type"]
    if kind == "bool":
        return Bool(d["key"], d["question"])
    if kind == "choice":
        return Choice(d["key"], d["question"], options=list(d["options"]))
    if kind == "score":
        return Score(d["key"], d["question"], legend=list(d["legend"]))
    raise ValueError(f"unknown question type {kind!r}")


def answer_labels(q: Question) -> list[str]:
    """The answer names in the order probabilities come back in."""
    if isinstance(q, Bool):
        return ["true", "false"]
    if isinstance(q, Choice):
        return list(q.options)
    return [str(i) for i in range(len(q.legend))]
