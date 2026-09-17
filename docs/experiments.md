# Experiments log

Hardware unless noted: Apple M5 Pro, 24 GB unified memory, macOS. torch 2.14 (MPS), transformers 5.17,
llama.cpp via Homebrew (ggml 0.10.1), Ollama 0.33.3.

## 2026-09-17 — engine correctness

Command: `.venv/bin/python -m pytest -q tests`

- Qwen2.5-1.5B-Instruct, float32: tree and batch strategies match naive per-question passes (<2e-3), both single-pass and one-branch-per-pass chunking. Prefix reuse hits P-1 tokens on repeat state.
- bf16 on MPS: up to 0.04 absolute drift on a near-50/50 boolean (0.518 vs 0.478). float32 agrees to 4 dp → precision, not masking.

## 2026-09-17 — smoke latency, 4 questions, ~100-token ticket state

Command: `.venv/bin/python scripts/smoke.py <spec>` (3 calls, same state; calls 2–3 are warm).

| backend | model | warm latency | notes |
|---|---|---|---|
| hf tree, bf16 MPS | Qwen2.5-1.5B-Instruct | ~59 ms | prefix fully cached on repeat |
| llamacpp, 8 slots | gemma-4-E2B-it UD-Q4_K_XL (Unsloth) | ~75–82 ms | exact grammar-restricted |
| ollama | qwen2.5:1.5b-instruct | ~100–120 ms | approximate top-20 |

Nuance: identical state repeated, so warm numbers are a best case for caching. Answers differ across
models (Gemma E2B: churn=4 at 0.998, revenue p=1.0 → overconfident; Qwen: revenue p≈0.25–0.52).
