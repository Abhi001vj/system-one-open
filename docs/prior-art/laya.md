# Laya (Convai Innovations) — notes and comparison

Sources, read 2026-10-02:

- GitHub <https://github.com/NandhaKishorM/laya> (`laya` 0.3.5 at time of reading)
- Models `convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions`
- Space <https://huggingface.co/spaces/convaiinnovations/laya-demo>
- Post: *"I built non-autoregressive decision models a year ago…"* (dev.to, 2026-09-18)
- Tracker: <https://huggingface.co/spaces/multimodalart/jev-decision-index>

Everything below is *their* claim or a dated public fact unless marked **ours**.

## What Laya is

A **trained 421M-parameter encoder specialist**, not a wrapper. ModernBERT-large (English) or
mmBERT-base (multilingual) encoder + a decision head: a question-type embedding, a `scorer`
MLP over `[MASK]` marker positions (one marker per option), and an `act_head` that predicts
act-vs-escalate from four features (top-1, top1−top2, entropy, k/255). Sequence layout is
`[CLS] instructions [MASK] opt1 [MASK] opt2 … [SEP] state [SEP]`; probabilities are
`softmax(logits / T)` with `T` fitted per (question type, option count) after training and
clamped to [0.5, 5.0]. Primitives are `noul` / `choice` / `score` — the same three as Jev.

Training ("RLCD" as implemented in their Kaggle notebook): sample G=4 noisy logit
distributions, score them with a strictly proper rule (log + spherical + ranked probability),
subtract a mean-reward baseline for the advantage, and add cross-entropy to the soft targets at
full weight (`loss_rl + 1.0 * loss_ce`). 2×T4, 4 epochs, ~4–5 h. Apache 2.0.

A `Router` picks a checkpoint per request by script/language detection *before* the forward
pass, because the English checkpoint collapses on non-Latin script while staying confident
(Khmer: 0.000 accuracy at 0.952 confidence — their number).

## Dates: Laya is not prior art for Jev, but its author has older work

- Jev early access: 2026-09-15.
- `laya` GitHub repo created **2026-09-18T04:46Z**; PyPI 0.1.0 **2026-09-18T04:38Z**; the model
  repo **2026-09-18T05:05Z**; the first commit in our clone is 2026-09-18 10:16 IST and the
  "Laya vs TypeSafe Jev" comparison landed 40 minutes later the same morning.
- The genuinely older artifacts are a different problem: **SalesRLAgent**
  (arXiv:2503.23303, 2025-03-30) — PPO over frozen sequence embeddings predicting one
  sales-conversion probability — and `DeepMostInnovations/sales-conversion-model-reinf-learning`
  (2025-05-11). A single vertical regressor, not arbitrary typed schemas over shared state.
- Reaction is split and partly hostile (HN 2026-09-19, ~1.3k points: dates, target leakage in
  SalesRLAgent, "won the benchmark by fine-tuning on it").

**ours:** so the honest framing for this repo does not change. Neither Jev nor Laya originated
reading answers from restricted answer-token logits. The ancestors to cite are PET
(arXiv:2001.07676), LM-BFF (arXiv:2012.15723), NLI zero-shot classification (arXiv:1909.00161),
GLiNER/GLiClass (arXiv:2311.08526), SetFit (arXiv:2209.11055), lm-evaluation-harness
`loglikelihood` scoring, and the calibration line (Guo 2017, Zhao 2021). What was new in Jev is
the product: calibration as a trained objective plus a typed API. See `resources.md`.

## How it compares to this repo

| | Laya | system-one-open (**ours**) |
|---|---|---|
| kind | trained 421M encoder + heads | inference engine over any open decoder LLM |
| weights | 3 Apache-2.0 checkpoints on the Hub | no checkpoint published yet (pilot LoRA only) |
| one pass for N questions | yes, batched sequences (one per question) | yes, tree attention (one packed sequence) or KV tiling |
| probabilities | learned head + fitted temperature | exact softmax over answer tokens; approximate only on Ollama |
| latency | 32.8–39.5 ms / 1 q, 6.8–7.2 ms/q batched (T4) | 59–120 ms / 4 q warm, 225 ms / 27 q (Qwen2.5-1.5B, M5 Pro MPS) |
| state budget | 320 tokens (English ckpt) / ~768 (multilingual) | the model's full context |
| option cap | ~20 in practice; 77 labels collapse to 0.425 on Banking77 (their number) | 52 letters today; no sequence-budget ceiling |
| multilingual | 45/51 languages usable via router | **none** — English prompts only, untested |
| zero-shot quality | base checkpoints 0.361/0.342 on typed-decisions, *below* the 0.461 majority baseline | unmeasured on any gold-label benchmark |
| specialising | RLCD fine-tune, 2×T4, 4–5 h | calibration LoRA to a teacher, 1 H100, $0.46 pilot |
| calibration | ECE 0.466 → 0.081 (English) by refitting temperature per bucket | ECE 0.236 → 0.040 via LoRA; beat best-temperature (0.676 KL) at 0.211 |

Not a like-for-like latency comparison: theirs is a T4, ours is a laptop GPU, and a 421M encoder
will stay faster per question than a 1.5B decoder. Speed is not where we compete.

## What their headline number actually says

Laya's README leads with 0.766 on `LocalLLaMA/typed-decisions` vs "Jev 0.727". Three caveats,
all visible in their own files:

1. **It is a fitted specialist.** 0.766 belongs to `laya-typed-decisions`, fine-tuned on that
   benchmark's own train split (1,200 cases / 6,000 items; the test split of 400 cases is held
   out, so this is not leakage — but it is a different category). The dataset's leaderboard
   scores "specialist, fitted per workflow" separately: ModernBERT-base gets 0.646, MiniLM-L6
   0.587. Their *zero-shot* checkpoints score 0.361 and 0.342, under the 0.470 input-ignoring
   prior.
2. **The Jev column was never measured by them** — no API access; it is copied from published
   third-party numbers, so prompts and sample sizes differ.
3. **Jev is also no longer the top line on that leaderboard.** meraGPT `sd-1` reports 0.768
   accuracy / 0.096 KL / 526 ms zero-shot (as of 2026-09-22).

Jev still leads Laya on soft accuracy (0.580 vs 0.471), raw ECE, option-order stability (0.13 vs
0.15–0.23) and high-cardinality choices.

## What this means for us

1. **We have no number on this benchmark at all, and that is now the gap that matters.**
   `LocalLLaMA/typed-decisions` is Apache-2.0, 400 test cases × 5 questions, with full gold
   distributions, created 2026-09-16, and its request body *is* our shape: `noul`→`Bool`,
   `choice`→`Choice` (option descriptions in `criteria` get rendered into the option line),
   `score`→`Score`. Accuracy, KL-to-gold and Brier are all computable from what `decide()`
   already returns. This is a small adapter, not a research project.
2. Report both modes, labelled: zero-shot (several models/backends) and after a calibration
   LoRA trained on the train split only. The second is the apples-to-apples row against
   `laya-typed-decisions`.
3. Their weak spots are our natural claims: no state-budget ceiling, no option-count cliff, and
   any model the user already runs. Their strong spots we should not contest: per-question
   latency, and multilingual coverage we do not have.
4. The engine has no `shortlist()` despite the error message in `types.py` pointing at one; Laya
   shipped `predict_shortlist`. Worth closing if we ever report a Banking77 row.
