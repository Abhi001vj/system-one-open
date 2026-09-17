# system-one-open

**Turn any open-weights model into a fast, typed, calibrated decision function.**

State in → many typed questions answered in parallel → schema-valid values with probabilities out.
No JSON generation, no parsing, no retries. Works with 1–4B models in-process on a laptop and with
larger models through llama.cpp or Ollama.

> Inspired by TypeSafe AI's *System One Models / Jev* announcement (Sep 2026). This is an
> independent, open re-implementation of the *interface idea* on top of ordinary LLMs — not
> affiliated with TypeSafe, and not a reproduction of their model or training (RLCD).
> See [`docs/prior-art/typesafe-jev.md`](docs/prior-art/typesafe-jev.md).

```python
from system_one import Decider, Bool, Choice, Score

d = Decider("qwen2.5-1.5b")            # or "qwen3.5-4b", "llamacpp", "ollama:gemma4:26b"
r = d.decide(ticket_text, [
    Bool("revenue_impacted", "Is revenue currently impacted?"),
    Choice("department", "Which department should own this?", options=["billing", "technical", "sales"]),
    Score("churn", "How likely is churn?", legend=["none", "low", "medium", "high", "certain"]),
])
r.answers["department"]   # {"choice": "technical", "confidence": 0.998, "probabilities": {...}}
```

## How it works

1. **Prefill once.** The system prompt + state is encoded and its KV cache is kept (and reused
   across calls up to the longest common token prefix).
2. **Branch per question.** Every question is a short suffix ending at the assistant turn.
3. **Score, don't generate.** The next-token logits at the end of each branch are restricted to
   that question's answer tokens (`yes/no`, option letters, score digits) and softmaxed.

Branches never see each other, so all of them run in **one forward pass**. The details depend on
the backend:

| backend | spec | probabilities | how branches run |
|---|---|---|---|
| transformers (MPS/CUDA/CPU) | `hf:<model id>` | exact | **tree attention** (one packed sequence, custom 4D mask) for pure-attention models; **batched KV tiling** for hybrid/recurrent (Qwen3.5, LFM2) |
| llama.cpp `llama-server` | `llamacpp:<url>` | exact (grammar-restricted, renormalized) | concurrent 1-token requests, batched across server slots |
| Ollama | `ollama:<tag>` | approximate (top-k logprobs) | concurrent 1-token requests |

Tests check that packed/batched scoring equals one independent forward pass per question
(float32, `tests/test_equivalence.py`).

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[doom]" pytest
.venv/bin/python -m pytest -q                       # equivalence tests
.venv/bin/python scripts/smoke.py qwen2.5-1.5b     # 4 questions, latency per call

# bigger models
llama-server -m <any>.gguf --port 8091 -np 16 -c 32768 --kv-unified --jinja
.venv/bin/python scripts/smoke.py llamacpp
.venv/bin/python scripts/smoke.py ollama:gemma4:26b
```

## Demos

```bash
s1-race --model qwen2.5-1.5b                     # 27 questions: parallel vs autoregressive JSON, same weights
s1-doom --model qwen2.5-1.5b                     # ViZDoom window + live judgement bars; --headless, --realtime, --order "..."
s1-wikirace --model llamacpp --all --baseline    # tournament link picking vs an LLM typing link titles
```

First results (M5 Pro 24 GB, details and caveats in [`docs/experiments.md`](docs/experiments.md)):

- **Race:** 27 typed answers in 225–425 ms vs 6.7 s for autoregressive JSON on the *same* Qwen2.5-1.5B (≈16×).
- **Doom:** two-stage decisions in ~110 ms in-process (≈9/s). Small models' judgement is the bottleneck, not speed.
- **Wikirace:** 0 hallucinated links by construction; the autoregressive 1.5B baseline hallucinated 18–30 titles per race.
- **Honest gap:** 1–2B models are poorly calibrated and phrasing-sensitive. Gemma-4-E4B is much better.
  Calibration fine-tuning is the next real step ([`docs/roadmap.md`](docs/roadmap.md)).

## Docs

- [`docs/README.md`](docs/README.md) — index
- [`docs/design.md`](docs/design.md) — engine design, backends, limits
- [`docs/experiments.md`](docs/experiments.md) — measured results log
- [`docs/resources.md`](docs/resources.md) — papers, libraries, links
- [`docs/plan.md`](docs/plan.md) — **next:** calibration LoRA, then shared-prefix llama.cpp backend
- [`docs/roadmap.md`](docs/roadmap.md) — demos, calibration fine-tuning, MLX backend
- [`results/`](results/) — raw run records cited in the experiments log
