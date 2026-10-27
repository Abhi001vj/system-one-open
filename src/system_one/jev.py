"""Jev (TypeSafe AI) through the Vercel AI Gateway, as a drop-in for `Decider`.

The gateway's evaluate API is TypeScript-only, so this talks to the Node sidecar in
`js/jev-server.ts`:

    cd js && node --env-file=../.env.local --experimental-strip-types jev-server.ts

Same `.decide(state, questions, preamble)` signature and the same answer dicts as the
local backends, so the demos and evaluation suites run against Jev unchanged. This makes
Jev the reference point the open-model wrapper is measured against.
"""

from __future__ import annotations

import time

import requests

from .decider import Result
from .types import Question, to_spec


class JevDecider:
    exact = True

    def __init__(self, url: str = "http://localhost:8099", model: str = "typesafe-ai/jev", timeout: float = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.http = requests.Session()
        self.timeout = timeout

    class _Backend:  # mimics the attributes suites read off a Decider
        def __init__(self, name: str):
            self.name = name
            self.exact = True

        def describe(self) -> str:
            return self.name

        def generate_stream(self, *a, **k):
            raise NotImplementedError("Jev does not generate text; it only answers typed questions")

    @property
    def backend(self):
        return JevDecider._Backend(f"jev:{self.model}")

    def decide(self, state: str, questions: list[Question], preamble: str = "") -> Result:
        t0 = time.perf_counter()
        # Jev takes one shared state; the preamble is prepended so it sees the same context.
        full_state = (preamble.strip() + "\n\n" if preamble else "") + state.strip()
        specs = [to_spec(q) for q in questions]
        forced = {q.key: q.interpret([1.0]) for q in questions if getattr(q, "options", None) and len(q.options) == 1}
        asked = [s for s in specs if s["key"] not in forced]
        answers, usage = {}, {}
        if asked:
            r = self.http.post(
                f"{self.url}/evaluate", json={"state": full_state, "questions": asked}, timeout=self.timeout
            )
            if r.status_code != 200:
                raise RuntimeError(f"jev sidecar error: {r.text[:300]}")
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(f"jev error: {payload['error'][:300]}")
            answers = payload["answers"]
            usage = payload.get("usage") or {}
        merged = {q.key: answers.get(q.key) or forced[q.key] for q in questions}
        return Result(
            merged, time.perf_counter() - t0, f"jev:{self.model}", True,
            {"forced": len(forced), "questions": len(asked), **{f"usage_{k}": v for k, v in usage.items() if v}},
        )
