# Design

## Interface

`Decider(backend).decide(state, questions, preamble="") -> Result`

Question types (`src/system_one/types.py`):

| type | prompt ends with | answer tokens | typed output |
|---|---|---|---|
| `Bool` | "Answer yes or no." | `yes/Yes/YES` vs `no/No/NO` (variants summed) | `p_true`, `value` |
| `Choice` | lettered options, "Answer with the letter" | `A`..`Z`, `a`..`z` (max 52) | `choice`, `confidence`, `probabilities` |
| `Score` | 0..N legend, "Answer with the single digit" | `0`..`9` | expected `score`, argmax `level`, `confidence` |

Output validity is by construction: the value is picked from a closed set, never parsed from text.

## Prompt layout

```
<chat template: system>                     ─┐
<user>  [preamble] STATE: <state>            ├─ shared prefix (KV cached, reused)
                                            ─┘
        \n\nQuestion: ... Options ... Answer…  ─┐ one branch per question
<end user><assistant start>                    ─┘ logits read at the last token
```

The chat template is split with an invisible marker, so any model's template works (Qwen, Gemma,
Llama…). Thinking is disabled via `enable_thinking=False`.

## HF backend strategies

**tree** (pure-attention models): all branch tokens packed into one sequence of length `Σ|branch|`.
Attention mask row for token *i* of branch *b*: all prefix columns + columns of branch *b* up to *i*.
Position ids restart at `P` for every branch, so each branch is positionally identical to running it
alone. After the pass the cache is cropped back to `P`.

**batch** (hybrid linear-attention / conv / sliding-window models): a custom mask is invalid for
recurrent layers, so the prefix cache is deep-copied, tiled to batch `B`, and branches run
right-padded. Right padding is safe: we only read each branch's last real token.

Chunking: `max_pass_tokens` bounds computed tokens per pass (tree: Σ lengths; batch: B × max length).

Prefix reuse: attention caches crop to the longest common token prefix with the previous call.
Recurrent state cannot be rewound, so hybrid models only reuse an identical prefix.

## Exactness

- `hf`: float32 matches naive per-question forward passes to <2e-3 (tests). bf16 on MPS drifts by
  up to ~0.04 on near-50/50 booleans — numeric noise, not a mask bug (verified in float32).
- `llamacpp`: GBNF grammar restricts to answer strings, `post_sampling_probs` with temperature 1 and
  no top-k/p returns the softmax renormalized over allowed tokens. Partial tokens (`y`, `ye`) are
  folded into their answer.
- `ollama`: top-20 logprobs only. Answers outside the top-k get a floor. Treat as approximate.

## Known limits

1. **Independence.** Branches can't condition on each other. Dependent decisions need stages
   (e.g. goal → movement), each stage still one pass.
2. **Cardinality.** Letter keys cap `Choice` at 52. For hundreds of options (Wikirace links) use a
   *tournament*: split into groups of ≤26, one `Choice` per group, all groups in one pass, top-2 of each
   advance, repeat until ≤k remain, final `Choice`. Measured better and cheaper than one `Bool` per
   option, because options compete inside a group and branch text is shared.
3. **Calibration is the base model's.** Instruct models are often overconfident on letters. Fixing
   that is a training problem (see roadmap), not an inference trick.
4. **Answer-letter bias.** Small models prefer `A`. Mitigations: option shuffling + averaging
   (costs one extra branch per permutation, still parallel).
5. **Server backends** recompute the prefix per slot unless the server's prompt cache hits.

## Patterns that proved useful in the demos

- **Code decides what is possible, the model ranks it.** Build option lists from state (Doom:
  only goals that are feasible; targets that exist). A `Choice` with one option is resolved by the
  `Decider` without a model call (`stats["forced"]`).
- **Stages for dependence.** Doom stage A (goal, target, item, fire) → plan sentence appended to the
  state → stage B (movement, dodge). Each stage is one pass.
- **Keep branch text short.** Branch tokens are the cost. Put shared instructions in the preamble
  (prefix) and leave only the varying part in the branch.
- **Phrase questions as observations, not permissions.** "Is an enemy in the crosshair, so the player
  should fire this instant?" works; "Should the trigger be held down?" gets near-zero probability from
  some instruct models. See `experiments.md`.
