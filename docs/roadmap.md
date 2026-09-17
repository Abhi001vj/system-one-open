# Roadmap

## Now
- [x] Typed questions: Bool, Choice, Score
- [x] HF backend: tree attention + batched KV tiling, prefix cache reuse, equivalence tests
- [x] llama.cpp server backend (exact, grammar-restricted) — any GGUF incl. Unsloth quants
- [x] Ollama backend (approximate, top-k logprobs)
- [ ] `s1-race`: 27 questions, parallel vs autoregressive JSON on the same weights, live terminal UI
- [ ] `s1-doom`: ViZDoom, structured state → staged judgements (goal → fire/dodge/move/aim) → buttons
- [ ] `s1-wikirace`: stage 1 score all links in parallel, stage 2 Choice over top-k; track hops/time/hallucinations

## Next
- [ ] MLX backend (batched KV tiling) — fastest path for 8–30B models on Apple silicon
- [ ] Option-order debiasing: permutation-averaged Choice
- [ ] Hybrid-model prefix snapshots (store recurrent state at the static preamble boundary)
- [ ] Workflow example: security-alert triage → disposition → containment → playbook

## Calibration training (the "RLCD-like" part)
Goal: a released LoRA/checkpoint whose probabilities are calibrated on System-One-shaped questions.
1. Generate (state, questions) from diverse templates and real text (tickets, alerts, game states).
2. Teacher probabilities: sample a frontier model N times or elicit distributions; keep disagreement.
3. Train the small model on answer-token distributions with a proper scoring rule
   (KL to teacher / Brier), branches packed with the same tree mask as inference.
4. Unsloth for LoRA on CUDA (Colab/Kaggle), export HF + GGUF, load via `hf:` or `llamacpp:`.
5. Report ECE/Brier and teacher agreement per task family in `experiments.md`, with held-out families.
