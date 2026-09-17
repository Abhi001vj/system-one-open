# Results

Raw run records referenced by [`docs/experiments.md`](../docs/experiments.md), one folder per date.
`runs/` stays git-ignored for scratch runs; copy a run here when a result is cited.

## 2026-09-17

| file | what |
|---|---|
| `race-qwen2.5-1.5b.json` | 27-question race, parallel vs autoregressive JSON, same weights |
| `race-qwen3.5-4b.json` | same race, Qwen3.5-4B (hf batch strategy) |
| `doom-qwen25-1.5b.jsonl` | Doom decisions (state, plan, answers, latency) — first version of the agent |
| `doom-gemma4-e2b-llamacpp.jsonl` | Doom, 300 decisions, Gemma-4-E2B Q4 via llama-server |
| `doom-gemma4-e4b-llamacpp.jsonl` | Doom, 300 decisions, Gemma-4-E4B Q4 via llama-server |
| `wikirace-qwen25-all.json` | Wikirace v1 (before body-only links / redirect fix), Qwen2.5-1.5B |
| `wikirace-qwen25-all-v2.json` | Wikirace v2, Qwen2.5-1.5B |
| `wikirace-gemma4-e4b-llamacpp.json` | Wikirace v2, Gemma-4-E4B Q4 via llama-server |
