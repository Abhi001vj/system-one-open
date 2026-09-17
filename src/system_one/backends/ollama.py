"""Ollama backend. APPROXIMATE probabilities.

Ollama cannot constrain the first token to a custom set or share a KV prefix
across branches, so each question is a concurrent 1-token chat request with
`top_logprobs`. Answer mass is summed from the returned top-k tokens and
renormalized; answers outside the top-k get a tiny floor. Good for trying
large models you already have pulled; use llamacpp/hf when calibration matters.

Concurrency is bounded by the server: set OLLAMA_NUM_PARALLEL (e.g. 8) before
`ollama serve` for real parallelism.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import requests

from .base import Backend, BranchSpec, ScoreOutput, group_token_probs, normalize


class OllamaBackend(Backend):
    exact = False

    def __init__(self, model: str, url: str = "http://localhost:11434", workers: int = 16, top_k: int = 20, timeout: float = 300):
        self.model = model
        self.url = url.rstrip("/")
        self.name = f"ollama:{model}"
        self.http = requests.Session()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.top_k = top_k
        self.timeout = timeout

    def describe(self) -> str:
        return f"{self.name} [top-{self.top_k} approx]"

    def _chat(self, system: str, user: str, **extra) -> requests.Response:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "think": False,
            "keep_alive": "30m",
            **extra,
        }
        return self.http.post(f"{self.url}/api/chat", json=body, timeout=self.timeout, stream=extra.get("stream", False))

    def _one(self, system: str, user: str, b: BranchSpec) -> list[float]:
        r = self._chat(system, user, stream=False, logprobs=True, top_logprobs=self.top_k, options={"num_predict": 1, "temperature": 0})
        r.raise_for_status()
        lp = r.json().get("logprobs") or []
        if not lp:
            return normalize([0.0] * len(b.answers))
        pairs = [(t["token"], math.exp(t["logprob"])) for t in lp[0]["top_logprobs"]]
        return normalize(group_token_probs(pairs, b.answers))

    def score(self, system: str, user_head: str, branches: list[BranchSpec]) -> ScoreOutput:
        t0 = time.perf_counter()
        futures = [self.pool.submit(self._one, system, user_head + "\n\n" + b.body, b) for b in branches]
        dists = [f.result() for f in futures]
        return ScoreOutput(dists, {"requests": len(branches), "wall_s": time.perf_counter() - t0})

    def generate_stream(self, system: str, user: str, max_tokens: int = 1024) -> Iterator[str]:
        with self._chat(system, user, stream=True, options={"num_predict": max_tokens, "temperature": 0}) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
