# Resources

## Source posts
- TypeSafe — Introducing System One Models & Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe — Lies, Damned Lies, and Benchmarks: https://typesafe.ai/blog/antibenchmaxxing

## Parallel / shared-prefix decoding
- SGLang (RadixAttention, `fork` / `select` primitives): https://arxiv.org/abs/2312.07104 · https://github.com/sgl-project/sglang
- vLLM / PagedAttention (prefix caching): https://arxiv.org/abs/2309.06180
- Hydragen — shared-prefix batched attention: https://arxiv.org/abs/2402.05099
- SpecInfer — tree attention for token trees: https://arxiv.org/abs/2305.09781
- Medusa — tree attention over candidate continuations: https://arxiv.org/abs/2401.10774

## Constrained / structured decoding
- Outlines: https://github.com/dottxt-ai/outlines
- XGrammar: https://github.com/mlc-ai/xgrammar
- guidance: https://github.com/guidance-ai/guidance
- llama.cpp GBNF grammars: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

## Calibration and LMs as classifiers
- On Calibration of Modern Neural Networks (temperature scaling, ECE): https://arxiv.org/abs/1706.04599
- Language Models (Mostly) Know What They Know: https://arxiv.org/abs/2207.05221
- Calibrate Before Use (contextual calibration, label bias): https://arxiv.org/abs/2102.09690
- Large Language Models Are Not Robust Multiple Choice Selectors (option-letter bias): https://arxiv.org/abs/2309.03882

## Agents and environments
- ViZDoom: https://github.com/Farama-Foundation/ViZDoom
- VPT — factorized keyboard/mouse action heads for Minecraft: https://arxiv.org/abs/2206.11795
- MediaWiki Action API (links, parse): https://www.mediawiki.org/wiki/API:Links

## Runtimes used here
- transformers: https://github.com/huggingface/transformers
- llama.cpp server API: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- Ollama API (logprobs): https://github.com/ollama/ollama/blob/main/docs/api.md
- Unsloth (LoRA fine-tuning, GGUF export): https://github.com/unslothai/unsloth
- MLX / mlx-lm: https://github.com/ml-explore/mlx-lm

## Evals
- The Leaderboard Illusion: https://arxiv.org/abs/2504.20879
