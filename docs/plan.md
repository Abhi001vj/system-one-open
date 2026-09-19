# Plan: calibration LoRA, then a shared-prefix llama.cpp backend

Written 2026-09-17, after the first demo round (see [experiments.md](experiments.md)).

## Where we are

Speed and validity work: 27 typed answers in 225–425 ms on a 1.5B model, zero invalid outputs by
construction, exact probabilities on `hf` and `llamacpp`. What does **not** work is the quality of the
probabilities themselves:

| symptom | evidence |
|---|---|
| phrasing sensitivity | Gemma-4-E2B: p(fire) = 0.00 for "should the trigger be held down?" vs 1.00 for "is an enemy in the crosshair?" on the same state |
| no discrimination in 1–2B models | Qwen3.5-2B: p(fire) 0.65 with an enemy ahead, 0.65 in an empty room |
| overconfidence | Gemma-4-E2B race: churn level 4 at 0.998, revenue p = 1.00 |
| inconsistency across formats | Qwen2.5-1.5B: parallel and autoregressive answers agree on 14/27 |
| slow servers | llama-server re-prefills the state per slot: 0.7–1.3 s per Doom decision, parallel Wikirace slower than autoregressive |

Two workstreams follow, in this order.

---

## Status (2026-09-19)

The pipeline is built and a pilot has run end to end on Modal: teacher labelling with
`google/gemma-4-31B-it` on an H100 (5 tasks/s, $0.22 per 1k tasks) and a pilot LoRA on
Qwen2.5-1.5B that cuts KL to the teacher from 1.24 to 0.21 and ECE from 0.236 to 0.040,
beating temperature scaling (0.676 KL) — including on a held-out family. Numbers, caveats and
the cost model are in [experiments.md](experiments.md). Remaining for C4/C5: bigger and less
correlated data, an episode-level doom split, a second teacher, paraphrase augmentation, a
bigger student, then release.

## Workstream 1 — calibration LoRA (the checkpoint worth releasing)

**Goal:** a LoRA adapter (and merged HF + GGUF weights) whose answer-token probabilities are calibrated
and stable under rephrasing and option reordering on System-One-shaped questions.

### 1.1 Data

Each training example is `(state, [questions])`, where every question has a target distribution over its
allowed answers.

| family | source | target distribution | held out? |
|---|---|---|---|
| support tickets | synthetic tickets + the 27-question schema, extended | teacher | train |
| security alerts | synthetic alerts + triage/containment questions (TypeSafe's example workflow shape) | teacher | train |
| Doom states | `runs/doom-*.jsonl` states (thousands already logged) + scripted situations | rules from game state where exact (enemy in crosshair, health < 30) + teacher for tactics | train |
| Wikirace choices | link groups from real pages | teacher | **held out** |
| text classification | banking77, ag_news, go_emotions (already in local HF cache) as `Choice`/`Bool` | gold label, smoothed by teacher | partly held out |
| NLI / factual yes-no | public datasets mapped to `Bool` | gold label | **held out** |

Augmentations, applied at data-build time:

- **Paraphrases:** 3–5 rewordings per question, all sharing one target. This fixes the fire-question problem.
- **Option shuffling:** permute option order and letters; targets are permuted to match. This fixes letter bias.
- **Distractor states:** minimal edits that should flip the answer, like removing the enemy or changing health.
  Each becomes a pair with an opposite target.

### 1.1a Infrastructure (built)

- `modal_app/label.py` — teacher labelling on Modal, sharded, writes to the `s1-data` volume,
  reports throughput and $ per 1k tasks. Teacher is scored with the inference engine itself.
- `modal_app/train.py` — LoRA training on Modal; prints base vs temperature-scaled vs trained,
  per family, and the run cost. `--push-to-hub` publishes the adapter.
- `training/packing.py` — tree-masked packing and the KL loss; `training/trainer.py` — training,
  evaluation (KL/Brier/ECE/agreement) and the temperature-scaling baseline.

### 1.2 Teacher distributions

- **Primary:** a strong model with probabilities from its logprobs where available. Otherwise sample K = 10 at
  temperature 1 and use the empirical distribution with add-0.5 smoothing.
- **Disagreement:** where two teachers disagree, train on their average instead of dropping the example.
  "Genuinely ambiguous" should come out as a spread distribution.
- Where the correct answer can be computed from the state (Doom geometry, gold labels), code overrides the teacher.
- Budget first: ~20k states × ~10 questions. Log cost and teacher identity in the dataset card.

### 1.3 Training

- **Loss:** KL(target ‖ student) over each question's allowed answer tokens, read at the branch's last
  token. That is exactly what inference reads. Optionally add a Brier term. No loss on any other token.
- **Packing:** reuse the inference layout, so one sequence holds the state plus all its question branches
  under the tree mask, with loss at N positions. Cost per example is about one forward/backward pass
  regardless of question count.
- **LoRA:** r = 16, α = 32, on attention and MLP projections; 1–2 epochs; lr 1e-4 with cosine decay.
- **Base models, in order:**
  1. Qwen2.5-1.5B-Instruct: fast tree path on the Mac, cheapest iteration.
  2. Gemma-4-E4B-it: best local quality so far.
  3. Qwen3.5-2B/4B: shipped as GGUF, since the HF path is slow on MPS.
- **Hardware:**
  - Unsloth on Colab/Kaggle GPUs (L4/T4/A100) for the real runs.
  - Plain transformers + PEFT is the fallback if Unsloth's fused kernels reject a custom 4D mask. The
    other fallback is the batch layout, which needs no custom mask. Check this first (task 1.3a).
  - `mlx-lm` LoRA on the Mac for small smoke runs.

### 1.4 Evaluation

The adapter must beat **the base model + temperature scaling**, a one-parameter post-hoc fix.
Otherwise the LoRA isn't earning its keep.

| metric | what it catches |
|---|---|
| ECE (15 bins), Brier, NLL | calibration |
| argmax agreement with teacher / gold | accuracy |
| paraphrase spread: std of p across rewordings | phrasing sensitivity |
| permutation consistency: argmax stable under option shuffles | letter bias |
| demo metrics: fire-probe discrimination, race self-agreement (27 questions), Wikirace finish rate and hops, Doom kills/deaths on `defend_the_center` | does it matter end to end |

All results are reported **per held-out family**, following the rules in
[prior-art/antibenchmaxxing.md](prior-art/antibenchmaxxing.md): dated, with commands and nuance,
including regressions.

### 1.5 Release

- Hugging Face Hub: LoRA adapter, merged fp16 weights, and a GGUF Q4_K_M (via Unsloth or llama.cpp `convert`).
- Model card: intended use (typed decisions, not chat), eval tables per family, known failures,
  teacher disclosure, licence inherited from the base model.
- Presets in `src/system_one/backends/__init__.py` pointing at the released weights.

### Milestones

| id | deliverable | exit criterion |
|---|---|---|
| C1 | `training/` data builder + dataset card; 2k-example pilot | **done** — 1,487 tasks / 8,729 questions, one command |
| C2 | loss + packed training loop; PEFT + custom 4D mask | **done** — `training/packing.py`, `training/trainer.py`, runs locally and on Modal |
| C3 | pilot LoRA on Qwen2.5-1.5B | **done** — beats temperature scaling on every family incl. held-out (see experiments) |
| C4 | full run on Gemma-4-E4B | fire-probe discrimination ≥0.9 on all 4 phrasings; race self-agreement ≥24/27 |
| C5 | release on HF Hub (adapter + merged + GGUF) + model card | loads with `hf:` and `llamacpp:` presets and passes equivalence tests |

---

## Workstream 2 — shared-prefix llama.cpp backend (`gguf:`)

**Goal:** run any GGUF with the same economics as the `hf` tree path: prefill the state once and
branch every question from it inside one `llama_decode`.

### 2.1 Design

Bind to the llama.cpp C API with `ctypes`, against the installed `libllama.dylib` (Homebrew ships
`llama.h` with `llama_memory_seq_cp`, `llama_batch_init` and `llama_get_logits_ith`). If build friction is
low, `llama-cpp-python`'s low-level module is the alternative.

Per `decide()` call:

1. Tokenize prefix and branches with the model's own vocab (`llama_tokenize`).
2. **Prefix reuse:** find the longest common prefix with sequence 0, `llama_memory_seq_rm(mem, 0, common, -1)`,
   and decode only the new prefix tokens.
3. **Fork:** for each branch *i*, `llama_memory_seq_cp(mem, 0, i + 1, -1, -1)`. This shares KV cells;
   it doesn't copy tensors.
4. **One batch:** all branch tokens with `seq_id = i + 1`, positions starting at P, and `logits = true`
   only on each branch's last token. One `llama_decode`.
5. Read `llama_get_logits_ith` per branch, restrict to answer token ids, softmax.
6. **Clean up:** `llama_memory_seq_rm(mem, i + 1, -1, -1)` for every branch.

Context settings: `n_seq_max` ≥ max branches per pass + 1, unified KV, `n_batch` ≥ branch tokens per pass.
Chunk branches exactly like `max_pass_tokens`.

Chat templates: the GGUF's embedded jinja template isn't fully supported by `llama_chat_apply_template`.
Use the HF tokenizer's template when a repo id is given, the built-in template otherwise, and allow a
`--template-from` override.

### 2.2 Risks to check first

- **Sliding-window attention (Gemma):** does `seq_cp` interact with the SWA cache? It may need
  `swa_full = true`.
- **Recurrent / hybrid models (Qwen3.5):** llama.cpp supports `seq_cp` for recurrent state. Check that
  branch outputs match independent runs.
- **ABI churn:** the C API changes between ggml releases. Pin a version at load time, fail with a clear
  message, and keep the `llamacpp:` server backend as the stable fallback.

### Milestones

| id | deliverable | exit criterion |
|---|---|---|
| G1 | ctypes binding: load model, tokenize, decode prefix, logits | smoke test passes on Gemma-4-E2B Q4 |
| G2 | fork/batch/cleanup scoring + prefix reuse | probabilities match the `llamacpp:` server backend within 5e-3 on the equivalence questions for Gemma-4-E2B, Gemma-4-E4B and Qwen3.5-2B |
| G3 | benchmarks in `experiments.md` | Doom decision with Gemma-4-E4B < 250 ms; parallel Wikirace wall time < autoregressive on the same model; race 27 questions < 400 ms |
| G4 | presets for the local Unsloth GGUFs (Gemma-4 E2B/E4B/26B-A4B, Qwen3.6-27B) | 26B-A4B runs the race on the 24 GB Mac |

---

## Also queued (smaller)

- Doom on `defend_the_center` / `deadly_corridor` with `--realtime` numbers per model.
- Web UI (frame + judgement bars + compute graph) and README GIFs.
- MLX backend with batched KV tiling.
- Permutation-averaged `Choice` as an inference-time letter-bias fix, which also gives an extra baseline for C3.
