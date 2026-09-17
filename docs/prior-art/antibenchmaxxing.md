# TypeSafe AI — "Lies, Damned Lies, and Benchmarks"

Source: https://typesafe.ai/blog/antibenchmaxxing. Notes taken 17 Sep 2026 (summary, not verbatim
except quotes).

## Argument

- Benchmarks drove early progress, but optimizing for them ("benchmaxxing") produces jagged
  capability: "Post-training therefore pushes capability fastest in the illuminated areas, while
  reliability outside them can stay flat or even get worse." (streetlight effect)
- Examples cited: Llama 4's private Arena variants (see *The Leaderboard Illusion*,
  https://arxiv.org/abs/2504.20879); an agent on a vending-machine benchmark forming price cartels
  and lying to suppliers (Andon Labs); an intelligence index revised shortly after a launch;
  a leaderboard win driven by templated, longer output.
- Unspecialized humans scored 34.5% on MMLU (https://arxiv.org/abs/2009.03300), yet superhuman
  benchmark scores haven't translated into proportional economic impact.

## Their recommendations

- Labs: publish caveats, disclose cherry-picking, include evidence that makes you look bad,
  de-emphasize benchmarks even when ahead.
- Users: run private evals, treat public ones with suspicion, don't amplify every plot.
- TypeSafe: no standard benchmark tables; dated one-off eval snapshots that are retired, not
  hill-climbed.

## Rules we adopt for this repo (ours)

1. Every number in `experiments.md` has date, hardware, backend spec, model and command.
2. Each result lists the nuance that flatters us (short state, easy questions, warm cache).
3. Baselines use the same model where possible (parallel vs autoregressive on identical weights)
   so the comparison isolates the decoding method, not model quality.
4. Report agreement with a stronger teacher and calibration (Brier, ECE), not just accuracy.
5. Keep failures and negative results.
