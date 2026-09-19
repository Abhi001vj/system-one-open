"""Hugging Face transformers backend (MPS / CUDA / CPU). Exact probabilities.

Two strategies, picked from the model architecture:

* tree  — pure-attention models. Branches are packed into one sequence with a
          tree-shaped attention mask: each branch sees the prefix and itself.
          All questions cost one forward pass over the branch tokens only.
* batch — hybrid / recurrent models (Qwen3.5 Gated DeltaNet, LFM2 convs) and
          sliding-window models, where a custom mask is wrong. The prefix
          cache is tiled across the batch and branches run right-padded.

The prefix KV cache is kept between calls and reused up to the longest common
token prefix (attention-only models; recurrent state cannot be rewound).
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, TextIteratorStreamer

from .base import MARK, Backend, BranchSpec, ScoreOutput

_ATTENTION_LAYERS = {"full_attention", "attention"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_id: str, dtype, device_map):
    """Load a decoder, whatever head class the checkpoint declares."""
    kwargs = dict(dtype=dtype, attn_implementation="sdpa")
    if device_map:
        kwargs["device_map"] = device_map
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def _tile(value, n: int):
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(n, dim=0) if value.dim() > 0 and value.shape[0] == 1 else value
    if isinstance(value, list):
        return [_tile(v, n) for v in value]
    if isinstance(value, dict):
        return {k: _tile(v, n) for k, v in value.items()}
    return value


def _tile_cache(cache: DynamicCache, n: int) -> None:
    """Repeat every per-sequence state of every cache layer along the batch axis.

    Works for attention KV and for linear-attention conv/recurrent states, which
    don't all implement `batch_repeat_interleave`.
    """
    for layer in cache.layers:
        for name, value in list(vars(layer).items()):
            tiled = _tile(value, n)
            if tiled is not value:
                setattr(layer, name, tiled)


class HFBackend(Backend):
    exact = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        strategy: str = "auto",
        max_pass_tokens: int = 3072,
        device_map: str | None = None,
    ):
        self.device = device or pick_device()
        if dtype is None:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model_id = model_id
        self.name = f"hf:{model_id}"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = _load_model(model_id, dtype, device_map)
        if device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        # multimodal checkpoints (Gemma 4, Qwen3.5) wrap the decoder one level deeper
        self.body = getattr(self.model.model, "language_model", None) or self.model.model
        self.head = self.model.get_output_embeddings()
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        layer_types = set(getattr(cfg, "layer_types", None) or ["full_attention"])
        self.pure_attention = layer_types <= _ATTENTION_LAYERS
        self.strategy = strategy if strategy != "auto" else ("tree" if self.pure_attention else "batch")
        self.max_pass_tokens = max_pass_tokens
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._cache: DynamicCache | None = None
        self._cache_ids: list[int] = []
        self._first_tok: dict[str, int] = {}

    def describe(self) -> str:
        return f"{self.name} [{self.strategy}, {self.device}]"

    # ------------------------------------------------------------------ prompt
    def render(self, system: str, user: str, generation_prompt: bool = True) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False
        )

    def split_template(self, system: str, user_head: str) -> tuple[str, str]:
        head, tail = self.render(system, user_head + MARK).split(MARK)
        return head, tail

    def _answer_ids(self, variants: list[str]) -> list[int]:
        out = set()
        for v in variants:
            if v not in self._first_tok:
                self._first_tok[v] = self.tok.encode(v, add_special_tokens=False)[0]
            out.add(self._first_tok[v])
        return sorted(out)

    # ----------------------------------------------------------------- caching
    def _truncate(self, length: int) -> None:
        extra = self._cache.get_seq_length() - length
        if extra > 0:
            self._cache.crop(-extra)

    @torch.inference_mode()
    def _prefill(self, prefix_ids: list[int]) -> int:
        common = 0
        if self._cache is not None:
            n = min(len(prefix_ids), len(self._cache_ids))
            while common < n and prefix_ids[common] == self._cache_ids[common]:
                common += 1
            if self.pure_attention:
                common = min(common, len(prefix_ids) - 1)
                self._truncate(common)
            elif common == len(prefix_ids) == len(self._cache_ids):
                return common  # identical prefix: recurrent state is still valid
            else:
                common = 0
                self._cache = None
        if self._cache is None:
            self._cache = DynamicCache(config=self.model.config)
        new = prefix_ids[common:]
        self.body(
            input_ids=torch.tensor([new], device=self.device),
            position_ids=torch.arange(common, len(prefix_ids), device=self.device)[None],
            past_key_values=self._cache,
            use_cache=True,
        )
        self._cache_ids = list(prefix_ids)
        return common

    # ----------------------------------------------------------------- scoring
    def _logp(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.log_softmax(logits, dim=-1)

    @torch.inference_mode()
    def _tree_pass(self, chunk: list[list[int]], P: int) -> torch.Tensor:
        ids, pos, owner, last = [], [], [], []
        for i, b in enumerate(chunk):
            ids += b
            pos += range(P, P + len(b))
            owner += [i] * len(b)
            last.append(len(ids) - 1)
        dev, Q = self.device, len(ids)
        owner_t = torch.tensor(owner, device=dev)
        local = torch.arange(Q, device=dev)
        tree = (owner_t[:, None] == owner_t[None, :]) & (local[:, None] >= local[None, :])
        mask = torch.cat([torch.ones(Q, P, dtype=torch.bool, device=dev), tree], dim=1)[None, None]
        out = self.body(
            input_ids=torch.tensor([ids], device=dev),
            position_ids=torch.tensor([pos], device=dev),
            attention_mask=mask,
            past_key_values=self._cache,
            use_cache=True,
        )
        self._truncate(P)
        return self._logp(out.last_hidden_state[0, torch.tensor(last, device=dev)])

    @torch.inference_mode()
    def _batch_pass(self, chunk: list[list[int]], P: int) -> torch.Tensor:
        B, L, dev = len(chunk), max(map(len, chunk)), self.device
        ids = torch.full((B, L), self.pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros((B, P + L), dtype=torch.long, device=dev)
        attn[:, :P] = 1
        for i, b in enumerate(chunk):
            ids[i, : len(b)] = torch.tensor(b, device=dev)
            attn[i, P : P + len(b)] = 1
        cache = copy.deepcopy(self._cache)
        _tile_cache(cache, B)
        out = self.body(
            input_ids=ids,
            attention_mask=attn,
            position_ids=torch.arange(P, P + L, device=dev)[None].expand(B, L),
            past_key_values=cache,
            use_cache=True,
        )
        last = torch.tensor([len(b) - 1 for b in chunk], device=dev)
        return self._logp(out.last_hidden_state[torch.arange(B, device=dev), last])

    def score(self, system: str, user_head: str, branches: list[BranchSpec]) -> ScoreOutput:
        t0 = time.perf_counter()
        head, tail = self.split_template(system, user_head)
        prefix_ids = self.tok.encode(head, add_special_tokens=False)
        branch_ids = [self.tok.encode("\n\n" + b.body + tail, add_special_tokens=False) for b in branches]
        answer_ids = [[self._answer_ids(v) for v in b.answers] for b in branches]
        for b, groups in zip(branches, answer_ids):
            flat = [t for g in groups for t in g]
            if len(flat) != len(set(flat)):
                raise ValueError(f"answers share a first token: {b.answers}")

        reused = self._prefill(prefix_ids)
        self._sync()
        t_prefill = time.perf_counter()

        P = len(prefix_ids)
        run = self._tree_pass if self.strategy == "tree" else self._batch_pass
        def cost(chunk: list[list[int]]) -> int:  # tokens the pass will actually compute
            if self.strategy == "tree":
                return sum(map(len, chunk))
            return len(chunk) * max(map(len, chunk))  # right-padded batch

        logps, chunk, passes = [], [], 0
        for ids in branch_ids:
            if chunk and cost(chunk + [ids]) > self.max_pass_tokens:
                logps.append(run(chunk, P))
                passes += 1
                chunk = []
            chunk.append(ids)
        if chunk:
            logps.append(run(chunk, P))
            passes += 1
        logp = torch.cat(logps)

        dists = []
        for i, groups in enumerate(answer_ids):
            s = torch.stack([torch.logsumexp(logp[i, torch.tensor(g, device=self.device)], 0) for g in groups])
            dists.append(torch.softmax(s, 0).cpu().tolist())
        t_end = time.perf_counter()
        return ScoreOutput(
            dists,
            {
                "prefix_tokens": P,
                "reused_tokens": reused,
                "branch_tokens": sum(map(len, branch_ids)),
                "passes": passes,
                "prefill_s": t_prefill - t0,
                "branches_s": t_end - t_prefill,
            },
        )

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    # --------------------------------------------------------- reference/naive
    @torch.inference_mode()
    def naive_score(self, system: str, user_head: str, branches: list[BranchSpec]) -> list[list[float]]:
        """One full, independent forward pass per question. For tests only."""
        head, tail = self.split_template(system, user_head)
        prefix_ids = self.tok.encode(head, add_special_tokens=False)
        out = []
        for b in branches:
            ids = prefix_ids + self.tok.encode("\n\n" + b.body + tail, add_special_tokens=False)
            h = self.body(input_ids=torch.tensor([ids], device=self.device)).last_hidden_state[0, -1]
            logp = self._logp(h[None])[0]
            s = torch.stack([torch.logsumexp(logp[torch.tensor(self._answer_ids(v), device=self.device)], 0) for v in b.answers])
            out.append(torch.softmax(s, 0).cpu().tolist())
        return out

    # ------------------------------------------------------------ generation
    def generate_stream(self, system: str, user: str, max_tokens: int = 1024) -> Iterator[str]:
        prompt = self.render(system, user)
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False, streamer=streamer)
        thread = threading.Thread(target=lambda: self.model.generate(**kwargs), daemon=True)
        thread.start()
        yield from streamer
        thread.join()
