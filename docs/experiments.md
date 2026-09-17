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

## 2026-09-17 — race demo: 27 questions, same weights both sides

Command: `s1-race --model qwen2.5-1.5b --save runs/race-qwen2.5-1.5b.json` (sequential, not concurrent).

| side | latency | notes |
|---|---|---|
| parallel (hf tree, bf16) | 425 ms end-to-end in the demo; 225 ms in isolated benchmark with cold prefix | 263 prefix + 847 branch tokens, 1 pass |
| autoregressive JSON, same model | 6,744 ms (first token 1,302 ms) | valid JSON, 0 schema errors this run |

Speedup 15.9× on identical weights. **Agreement 14/27** between the two decoding modes on the same
model — a 1.5B model is not self-consistent across formats. Several parallel answers are plainly wrong
(`revenue_impacted` p=0.04, `security_risk` level 3 at 0.99). Speed is solved; quality needs a bigger or
better-calibrated model.

Strategy benchmark (27 questions, cold prefix every call, 4 repeats, warm kernels):

| strategy | dtype | latency | prefill / branches |
|---|---|---|---|
| tree | bf16 | 225 ms | 51 / 175 ms |
| tree | fp16 | 229 ms | 52 / 178 ms |
| batch | bf16 | 310 ms | 51 / 256 ms |
| batch | fp16 | 309 ms | 51 / 257 ms |

Branch throughput equals prefill throughput (~5k tok/s for 1.5B on MPS): the custom mask does not
fall off a slow path; cost is simply branch tokens. Lever: shorter branch text (move shared
instructions into the prefix).

## 2026-09-17 — Doom (ViZDoom deathmatch.cfg, seed 7, 300 decisions, sync mode, 4 tics/decision)

Command: `s1-doom --model <spec> --headless --decisions 300 --seed 7 --log runs/doom-*.jsonl`,
summary via `scripts/doom_summary.py`.

| model / backend | median decision (2 stages) | goals chosen | fired | kills |
|---|---|---|---|---|
| Qwen2.5-1.5B hf tree | 111 ms (first version) | collect_health 92% at full health | 0 | 0 |
| Gemma-4-E2B Q4 llamacpp (8 slots) | 694 ms | collect_health 85% | 10 | 0 |
| Gemma-4-E4B Q4 llamacpp (16 slots) | 1,322 ms | collect_ammo_or_weapon 83% | 5 | 0 |

Findings:
1. ~9 decisions/s is reachable in-process with a 1.5B model (111 ms for two stages), matching the
   "10 queries/s" framing — but the small model's judgement is poor.
2. **Prompt phrasing dominates small-model decisions** (`scripts/probe_fire.py`, clear shot vs empty room):

   | question | E2B clear / empty | E4B clear / empty | Qwen2.5-1.5B clear / empty | Qwen3.5-2B clear / empty |
   |---|---|---|---|---|
   | "Should the player's trigger be held down right now?" | 0.00 / 0.00 | 1.00 / 0.00 | 0.49 / 0.71 | 0.65 / 0.65 |
   | "Should the player shoot right now?" | 0.01 / 0.04 | 0.05 / 0.00 | 0.03 / 0.04 | 0.60 / 0.71 |
   | "Is an enemy in the crosshair, so the player should fire this instant?" | **1.00 / 0.00** | **1.00 / 0.00** | 0.02 / 0.01 | 0.66 / 0.57 |
   | "Does the player have a clear shot at an enemy right now?" | 0.00 / 0.00 | 1.00 / 0.00 | 0.06 / 0.02 | 0.68 / 0.15 |

   Gemma E4B discriminates on 3/4 phrasings; the 1–2B models barely discriminate at all. This is the
   strongest argument for calibration training (roadmap) rather than prompt tweaking.
3. llama-server is slow for this workload: each parallel slot re-prefills the shared state, so cost
   is N × (prefix + branch). A true shared-prefix GGUF backend (llama.cpp `seq_cp`) is on the roadmap.
4. Agent-level play is weak in `deathmatch` (item-heavy map, enemies rarely in view): 0 kills for all.
   Code-side fixes applied since: feasible-goal filtering, out-of-view enemies, stuck/explore
   controller, fire phrasing. Needs a rerun and a combat-focused scenario.

Nuance: sync mode pauses the game while the model thinks, so latency does not hurt play here;
`--realtime` would.

## 2026-09-17 — Wikirace v1 (Qwen2.5-1.5B hf tree, 4 challenges, max 10 hops, ≤300 links/page)

Command: `s1-wikirace --model qwen2.5-1.5b --all --baseline --max-hops 10`.

| challenge | parallel: finished / hops / model time / halluc. | autoregressive: finished / hops / model time / halluc. |
|---|---|---|
| Rubber duck → Roman Empire | ✓ / 7 / 3.3 s / 0 | ✗ / 10 / 15.4 s / 18 |
| K-pop → Knuth–Morris–Pratt | ✗ / 10 / 8.2 s / 0 | ✗ / 10 / 24.0 s / 27 |
| Banana → Quantum mechanics | ✗ / 10 / 9.0 s / 0 | ✗ / 10 / 23.7 s / 24 |
| Taylor Swift → Photosynthesis | ✗ / 10 / 13.2 s / 0 | ✗ / 10 / 23.3 s / 30 |

- The first parallel design (one yes/no per link) was *slower* than autoregressive: 5.5 s for 73 links,
  weak ranking (all p_true < 0.1). Replaced with **tournament groups** (≤26 lettered options per
  branch, top-2 advance): ~1–1.5 s per hop at 300 links.
- Autoregressive 1.5B hallucinated a link title on almost every attempt (3 retries then random), so its
  path is mostly random walk. Parallel never hallucinates by construction.
- Bugs found and fixed after this run: links from references/navboxes ("Cf.", "Treccani") were
  candidates; redirect titles caused a revisit loop (Traditional Vietnamese dance ↔ Dance in Vietnam).
- Nuance: autoregressive side gets the same ≤300 links; with a stronger model it hallucinates far
  less, so the halluc. gap here overstates what you'd see vs frontier LLMs.

## 2026-09-17 — Wikirace v2 (body-only links, redirect fix, tournament groups)

Command: `s1-wikirace --model <spec> --all --baseline --max-hops 10`.

**Gemma-4-E4B-it UD-Q4_K_XL (Unsloth GGUF) via llama-server, 16 slots**

| challenge | parallel: fin / hops / model s / halluc | autoregressive: fin / hops / model s / halluc |
|---|---|---|
| Rubber duck → Roman Empire | ✓ / 2 / 4.7 / 0 (via Western World) | ✓ / 3 / 0.9 / 0 (via Trier) |
| K-pop → Knuth–Morris–Pratt | ✓ / 5 / 13.8 / 0 (AI → CS → Algorithm → Search algorithm) | ✓ / 10 / 11.5 / 3 |
| Banana → Quantum mechanics | ✓ / 4 / 10.8 / 0 (Musaceae → Plant → Light) | ✓ / 4 / 4.9 / 0 |
| Taylor Swift → Photosynthesis | ✓ / 7 / 19.6 / 0 | ✗ / 10 / 15.1 / 3 (stuck in Taylor Swift songs) |

Parallel 4/4 with ≤ hops; autoregressive 3/4. **But parallel is slower in wall time on llama-server**:
every group request re-prefills the shared state per slot, and a hop has several rounds. The autoregressive
agent only generates a few tokens, so it is cheap here. This is the benchmark that needs the shared-prefix
GGUF backend; don't quote a speedup from it.

**Qwen2.5-1.5B hf tree** — parallel 0/4 (4.5–10.6 s, 0 halluc), autoregressive 0/4 (12.3–27.6 s, 21–24
halluc). Too weak to navigate; parallel is 2–4× faster and never invalid, autoregressive is a random walk.

## 2026-09-17 — race demo with Qwen3.5-4B (hf batch strategy, bf16, MPS)

| side | latency |
|---|---|
| parallel | 5,703 ms (prefill 1,082 ms for 267 tokens; branches 4,618 ms for 995 tokens) |
| autoregressive JSON | 19,072 ms (first token 4,698 ms) |

Agreement 22/27 (vs 14/27 for Qwen2.5-1.5B): the bigger model is far more self-consistent. Latency is
bad because Qwen3.5's Gated DeltaNet layers have no fast MPS kernel in transformers (≈250 tok/s prefill).
**Use llama.cpp or MLX for Qwen3.5 on Apple silicon**; the HF batch path is for correctness and CUDA.
