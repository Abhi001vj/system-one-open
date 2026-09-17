# Roadmap

## Now
- [x] Typed questions: Bool, Choice, Score
- [x] HF backend: tree attention + batched KV tiling, prefix cache reuse, equivalence tests
- [x] llama.cpp server backend (exact, grammar-restricted) — any GGUF incl. Unsloth quants
- [x] Ollama backend (approximate, top-k logprobs)
- [x] `s1-race`: 27 questions, parallel vs autoregressive JSON on the same weights, live terminal UI
- [x] `s1-doom`: ViZDoom, structured state → staged judgements (goal → fire/dodge/move/aim) → buttons
- [x] `s1-wikirace`: tournament link selection; hops/time/hallucinations vs autoregressive baseline
- [ ] Doom: combat scenario (`defend_the_center`, `deadly_corridor`), realtime mode numbers, kills/deaths per model
- [ ] Web UI for demos (frame + judgement bars + compute graph), recorded GIFs for the README

## Next
- [ ] **Shared-prefix GGUF backend** via llama.cpp's C API (`llama_memory_seq_cp` + multi-sequence
      batch): prefill once, fork the KV per question. llama-server re-prefills per slot, measured
      0.7–1.3 s per Doom decision vs ~110 ms in-process.
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
