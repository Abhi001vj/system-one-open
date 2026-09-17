"""The public API: typed questions in, typed calibrated answers out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backends import DEFAULT_SYSTEM, Backend, BranchSpec, load_backend
from .types import Choice, Question


@dataclass
class Result:
    answers: dict[str, dict]
    latency_s: float
    backend: str
    exact: bool
    stats: dict = field(default_factory=dict)


class Decider:
    def __init__(self, backend: Backend | str = "qwen2.5-1.5b", system: str = DEFAULT_SYSTEM, **backend_kwargs):
        self.backend = load_backend(backend, **backend_kwargs) if isinstance(backend, str) else backend
        self.system = system

    @staticmethod
    def user_head(state: str, preamble: str = "") -> str:
        return (preamble.strip() + "\n\n" if preamble else "") + "STATE:\n" + state.strip()

    def decide(self, state: str, questions: list[Question], preamble: str = "") -> Result:
        keys = [q.key for q in questions]
        if len(keys) != len(set(keys)):
            raise ValueError("question keys must be unique")
        t0 = time.perf_counter()
        # a question with a single possible answer is decided by code, not the model
        forced = {q.key: q.interpret([1.0]) for q in questions if isinstance(q, Choice) and len(q.options) == 1}
        asked = [q for q in questions if q.key not in forced]
        answers, stats = {}, {"forced": len(forced)}
        if asked:
            specs = [BranchSpec(*q.render()) for q in asked]
            out = self.backend.score(self.system, self.user_head(state, preamble), specs)
            answers = {q.key: q.interpret(d) for q, d in zip(asked, out.dists)}
            stats.update(out.stats)
        answers = {q.key: answers.get(q.key) or forced[q.key] for q in questions}
        return Result(answers, time.perf_counter() - t0, self.backend.describe(), self.backend.exact, stats)
