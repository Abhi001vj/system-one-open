"""llama.cpp `llama-server` backend (any GGUF, e.g. Unsloth quants). Exact probabilities.

Each question is a 1-token completion constrained by a GBNF grammar to its
allowed answers, with `post_sampling_probs` so the returned distribution is the
softmax renormalized over exactly those tokens (temperature 1, no top-k/p).
Requests are sent concurrently; the server batches them across its parallel
slots (`-np`) and reuses cached prompt prefixes.

Start a server, e.g.:
    llama-server -m model.gguf --port 8091 -np 16 -c 32768 --kv-unified --jinja
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import requests

from .base import MARK, Backend, BranchSpec, ScoreOutput, group_token_probs, normalize


def _gbnf_literal(s: str) -> str:
    return json.dumps(s)  # JSON string escaping is valid GBNF literal syntax


class LlamaCppBackend(Backend):
    exact = True

    def __init__(self, url: str = "http://localhost:8091", workers: int = 32, timeout: float = 120):
        self.url = url.rstrip("/")
        self.name = f"llamacpp:{self.url}"
        self.http = requests.Session()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.timeout = timeout
        self._template_cache: dict[tuple[str, str], tuple[str, str]] = {}
        props = self.http.get(f"{self.url}/props", timeout=timeout).json()
        self.model_path = props.get("model_path", "?")
        self.slots = props.get("total_slots", 1)

    def describe(self) -> str:
        return f"llamacpp:{self.model_path.rsplit('/', 1)[-1]} [{self.slots} slots]"

    def _render(self, system: str, user: str) -> str:
        r = self.http.post(
            f"{self.url}/apply-template",
            json={
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["prompt"]

    def _split(self, system: str) -> tuple[str, str]:
        # the template only depends on the system prompt, so split once with the marker
        if system not in self._template_cache:
            head, tail = self._render(system, MARK).split(MARK)
            self._template_cache[system] = (head, tail)
        return self._template_cache[system]

    def _one(self, prompt: str, b: BranchSpec) -> list[float]:
        grammar = "root ::= " + " | ".join(_gbnf_literal(v) for group in b.answers for v in group)
        r = self.http.post(
            f"{self.url}/completion",
            json={
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": sum(len(g) for g in b.answers) * 3,
                "grammar": grammar,
                "post_sampling_probs": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "cache_prompt": True,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        top = r.json()["completion_probabilities"][0]["top_probs"]
        return normalize(group_token_probs([(t["token"], t["prob"]) for t in top], b.answers))

    def score(self, system: str, user_head: str, branches: list[BranchSpec]) -> ScoreOutput:
        t0 = time.perf_counter()
        head, tail = self._split(system)
        prefix = head + user_head
        futures = [self.pool.submit(self._one, prefix + "\n\n" + b.body + tail, b) for b in branches]
        dists = [f.result() for f in futures]
        return ScoreOutput(dists, {"requests": len(branches), "wall_s": time.perf_counter() - t0})

    def generate_stream(self, system: str, user: str, max_tokens: int = 1024) -> Iterator[str]:
        head, tail = self._split(system)
        with self.http.post(
            f"{self.url}/completion",
            json={"prompt": head + user + tail, "n_predict": max_tokens, "temperature": 0, "stream": True, "cache_prompt": True},
            stream=True,
            timeout=self.timeout,
        ) as r:
            for line in r.iter_lines():
                if line.startswith(b"data: "):
                    chunk = json.loads(line[6:])
                    if chunk.get("content"):
                        yield chunk["content"]
                    if chunk.get("stop"):
                        break
