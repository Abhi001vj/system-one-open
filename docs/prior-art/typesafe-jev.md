# TypeSafe AI — "Introducing System One Models & Jev" (15 Sep 2026)

Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev (Diogo Almeida, founder).
Notes taken 17 Sep 2026. Everything below is *their* claim unless marked **ours**.

## Claims

- New model class "System One Models": unstructured state in, typed probabilistic decisions out.
  Gives up string generation. "Can't hallucinate", no type errors (by construction).
- New architecture + parallel sampler + training method **RLCD** (Reinforcement Learning for
  Calibrated Decisions) optimizing for epistemically honest probabilities.
- Jev: "similar intelligence on System One tasks" to frontier LLMs, 40–200× faster, 70–500 ms
  end-to-end. Pricing $0.042 / MTok input, output free.
- Home-page numbers (193.6× faster, 444.6× cheaper) come from their *workflow evals*.
- "Jev is neither small nor an LLM."
- Choice cardinality up to 255; for higher cardinality they score independently then make an explicit choice (2 stages).

## Output types seen in their demo (27-question race)

- boolean: `{"noul": 0.85, "type": "noul"}` (probability of true)
- choice: `{"choice", "confidence", "probabilities"}`
- score: `{"score": 1.6, "confidence", "legend": {"0": ...}}` (expected value over levels)

**ours:** `Bool` / `Choice` / `Score` mirror these three.

## Evals and their stated nuance

- Side-by-side race: 27 questions, one request each, vs GPT-5.6 Terra. Short dense state (favours
  them, they say so). Only disagreement: churn likelihood.
- Workflow evals: reference = average of GPT-6 Astra and Fable 5.1 probabilities, fixed workflow code
  per model (no harness tuning). Biased towards those labs' models; workflows written by their team.
- Hallucination rates for LLMs measured via OpenRouter; theirs is 0% by construction, not measured.
- Example workflow (security alert): triage (3 readings) → code disposition (close/queue/act with
  thresholds, grey zone 0.15–0.60) → containment (11 readings: bools + scores + choices) →
  playbook (first applicable group, strongest action whose conditions hold) → escalate.

## Demos

- **Doom**: structured game state as text (not pixels, "yet"), ~10 queries/s (~$7/hour).
  Judgement panels: FIRING (bool), GOAL (choice, e.g. kill enemy / get item), DODGE (choice),
  MOVEMENT (choice), with a compute graph. Their caveat: a non-AI bot plays better; the point is
  reacting to state representations and following instructions.
- **Wikirace**: Jev vs GPT-5.6 Luna, Claude Sonnet 5, Claude Haiku 4.5 (non-reasoning modes).
  Tracks hops, time, cost and *hallucinated links*. Speedups smaller than in other demos.

## What we can and cannot replicate (ours)

| their component | open version here |
|---|---|
| parallel sampler | tree attention / batched KV tiling over an existing LLM — replicable today |
| typed outputs, zero type errors | replicable by construction |
| 70–500 ms latency | replicable for small models locally; big-model latency depends on hardware |
| calibration (RLCD) | **not** free — needs training. Plan: distill teacher probabilities + proper scoring rules (Brier/log loss) with LoRA; see roadmap |
| "not an LLM" architecture | unknown; out of scope |
| intelligence parity with frontier | not claimed; measure agreement vs a strong teacher per task |
