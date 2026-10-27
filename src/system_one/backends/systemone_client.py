"""Client for any `/v1/systemone` endpoint — Jev's own wire format.

Lets the demos and evaluation suites run unchanged against other implementations:
this repo's server, mmastrac/djev-spark (DiffusionGemma on vLLM), openjev, or a
hosted Jev-compatible service. Exposes the same `.decide()` as `Decider`, so the
same suite code measures every implementation.

    from system_one.backends.systemone_client import SystemOneClient
    d = SystemOneClient("http://localhost:8011")   # djev-spark
    d.decide(state, questions)
"""

from __future__ import annotations

import time

import requests

from ..decider import Result
from ..types import Bool, Choice, Question, Score


class SystemOneClient:
    exact = True

    def __init__(self, url: str = "http://localhost:8011", model: str = "jev-latest",
                 path: str = "/v1/systemone", timeout: float = 120, api_key: str = "",
                 extra: dict | None = None):
        self.url = url.rstrip("/") + path
        self.model = model
        self.http = requests.Session()
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.extra = extra or {}  # server extensions, e.g. {"samples": "auto", "think": 64}

    class _Backend:
        def __init__(self, name):
            self.name, self.exact = name, True

        def describe(self):
            return self.name

        def generate_stream(self, *a, **k):
            raise NotImplementedError("a /v1/systemone endpoint answers questions; it does not generate text")

    @property
    def backend(self):
        return SystemOneClient._Backend(f"systemone:{self.model}")

    @staticmethod
    def to_payload(questions: list[Question]) -> dict:
        out = {}
        for q in questions:
            if isinstance(q, Bool):
                out[q.key] = {"type": "noul", "instructions": q.question}
            elif isinstance(q, Choice):
                out[q.key] = {"type": "choice", "instructions": q.question,
                              "criteria": {o: None for o in q.options}}
            elif isinstance(q, Score):
                out[q.key] = {"type": "score", "instructions": q.question, "criteria": list(q.legend)}
        return out

    def decide(self, state: str, questions: list[Question], preamble: str = "") -> Result:
        t0 = time.perf_counter()
        forced = {q.key: q.interpret([1.0]) for q in questions
                  if isinstance(q, Choice) and len(q.options) == 1}
        asked = [q for q in questions if q.key not in forced]
        answers: dict[str, dict] = {}
        stats = {"forced": len(forced)}
        if asked:
            body = {"model": self.model, "state": state, "questions": self.to_payload(asked), **self.extra}
            if preamble:
                body["instructions"] = preamble
            r = self.http.post(self.url, json=body, timeout=self.timeout)
            if r.status_code != 200:
                raise RuntimeError(f"systemone endpoint {r.status_code}: {r.text[:300]}")
            payload = r.json()
            raw = payload.get("answers", {})
            for q in asked:
                a = raw.get(q.key)
                if a is None:
                    continue
                if isinstance(q, Bool):
                    p = float(a.get("noul", a.get("probability", 0.0)))
                    answers[q.key] = {"type": "bool", "p_true": p, "value": p >= 0.5}
                elif isinstance(q, Choice):
                    probs = a.get("probabilities") or {}
                    answers[q.key] = {
                        "type": "choice", "choice": a["choice"],
                        "confidence": float(a.get("confidence", probs.get(a["choice"], 1.0))),
                        "probabilities": {o: float(probs.get(o, 0.0)) for o in q.options},
                    }
                else:
                    probs = a.get("probabilities") or {}
                    levels = [float(probs.get(str(i), 0.0)) for i in range(len(q.legend))]
                    answers[q.key] = {
                        "type": "score", "score": float(a["score"]),
                        "level": max(range(len(levels)), key=levels.__getitem__) if any(levels) else round(float(a["score"])),
                        "confidence": float(a.get("confidence", max(levels) if levels else 1.0)),
                        "legend": a.get("legend") or dict(enumerate(q.legend)),
                    }
            usage = payload.get("usage") or {}
            stats.update({f"usage_{k}": v for k, v in usage.items() if v is not None})
            diag = payload.get("diagnostics") or {}
            if "latency_ms" in diag:
                stats["server_latency_ms"] = diag["latency_ms"]
        merged = {q.key: answers.get(q.key) or forced.get(q.key) for q in questions}
        missing = [k for k, v in merged.items() if v is None]
        if missing:
            raise RuntimeError(f"endpoint returned no answer for {missing}")
        return Result(merged, time.perf_counter() - t0, f"systemone:{self.model}", True, stats)
