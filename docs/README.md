# Docs index

| file | what |
|---|---|
| [design.md](design.md) | How the engine works, backend matrix, exactness guarantees, known limits |
| [experiments.md](experiments.md) | Dated log of measured results (latency, agreement, calibration) |
| [resources.md](resources.md) | Papers, libraries and links that this builds on |
| [plan.md](plan.md) | **Current plan:** calibration LoRA (checkpoint release) then shared-prefix llama.cpp backend, with milestones and exit criteria |
| [roadmap.md](roadmap.md) | Demos, calibration fine-tuning (Unsloth), MLX backend, open questions |
| [prior-art/typesafe-jev.md](prior-art/typesafe-jev.md) | Notes on TypeSafe's System One / Jev launch post: claims, nuance, what we can and can't replicate |
| [prior-art/antibenchmaxxing.md](prior-art/antibenchmaxxing.md) | Notes on TypeSafe's "Lies, Damned Lies, and Benchmarks" post and the eval rules we adopt from it |

Conventions: results go in `experiments.md` with date, hardware, backend spec and exact command.
Negative results are recorded too.
