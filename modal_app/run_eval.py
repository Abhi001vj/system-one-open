"""Head-to-head evaluation on Modal: wrapper vs autoregressive baseline, three suites.

One container per configuration (model, optional adapter), each running webnav / doom /
wikirace for both agents on the same weights. Results and per-step logs land on the
`s1-data` volume; cost is reported per configuration.

    modal run modal_app/run_eval.py --smoke                      # cheap plumbing check on L4
    modal run modal_app/run_eval.py --configs full                # the real matrix on H100
    modal volume get s1-data eval/ results/
"""

from __future__ import annotations

import json
import os
import time

import modal

GPU = os.environ.get("S1_GPU", "H100")
GPU_HOURLY = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
              "H100": 3.95, "H200": 4.54, "B200": 6.25, "B300": 7.10}

image = (
    modal.Image.from_registry("python:3.12-slim-bookworm", add_python=None)
    .apt_install("libsdl2-2.0-0", "libopenal1", "libgl1", "libgomp1", "libxext6", "libx11-6", "fontconfig",
               "build-essential")  # triton JIT needs a C compiler
    .pip_install("transformers>=5.17", "accelerate", "peft", "huggingface_hub", "rich", "requests",
                 "vizdoom>=1.3", "numpy", "beautifulsoup4")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir("src", "/repo/src", copy=True)
    .add_local_dir("eval", "/repo/eval", copy=True)
    .add_local_dir("training", "/repo/training", copy=True, ignore=["data", "out"])
    .add_local_file("pyproject.toml", "/repo/pyproject.toml", copy=True)
    .add_local_file("README.md", "/repo/README.md", copy=True)
    .run_commands("pip install --no-deps -e /repo")
)

app = modal.App("system-one-eval")
cache = modal.Volume.from_name("s1-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("s1-data", create_if_missing=True)


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/data": data}, timeout=6 * 60 * 60)
def evaluate(cfg: dict, gpu: str, run: str) -> dict:
    import sys

    sys.path.insert(0, "/repo")
    from eval.run import main

    t0 = time.perf_counter()
    out_dir = f"/data/eval/{run}"
    os.makedirs(out_dir, exist_ok=True)
    argv = [
        "--model", cfg["model"], "--label", cfg["label"], "--suites", cfg.get("suites", "webnav,doom,wikirace"),
        "--out", f"{out_dir}/{cfg['label'].replace('/', '_')}.json",
        "--log-dir", f"{out_dir}/logs-{cfg['label'].replace('/', '_')}",
        "--webnav-file", cfg.get("webnav_file", "/data/mind2web-train_10.json"),
        "--webnav-steps", str(cfg.get("webnav_steps", 100)),
        "--doom-seeds", cfg.get("doom_seeds", "1,2,3"),
        "--doom-decisions", str(cfg.get("doom_decisions", 200)),
        "--wiki-challenges", str(cfg.get("wiki_challenges", 8)),
    ]
    if cfg.get("adapter"):
        argv += ["--adapter", cfg["adapter"]]
    results = main(argv)
    elapsed = time.perf_counter() - t0
    price = GPU_HOURLY.get(gpu, 0)
    results["gpu"] = gpu
    results["cost_usd"] = round(elapsed / 3600 * price, 2)
    with open(f"{out_dir}/{cfg['label'].replace('/', '_')}.json", "w") as f:
        json.dump(results, f, indent=1)
    data.commit()
    print(f"{cfg['label']}: {elapsed / 60:.1f} min, ${results['cost_usd']:.2f}", flush=True)
    return results


SMOKE = [dict(label="qwen2.5-1.5b", model="Qwen/Qwen2.5-1.5B-Instruct", webnav_steps=6,
              doom_seeds="1", doom_decisions=10, wiki_challenges=1)]

FULL = [
    dict(label="gemma-4-31B-it", model="google/gemma-4-31B-it"),
    dict(label="gemma-4-E4B-it", model="google/gemma-4-E4B-it"),
    dict(label="qwen2.5-1.5b-base", model="Qwen/Qwen2.5-1.5B-Instruct"),
    dict(label="qwen2.5-1.5b-calib", model="Qwen/Qwen2.5-1.5B-Instruct",
         adapter="/data/adapters/pilot-qwen25-1.5b"),
]


@app.local_entrypoint()
def main(configs: str = "full", smoke: bool = False, run: str = "", webnav_steps: int = 0,
         doom_decisions: int = 0, wiki_challenges: int = -1, only: str = ""):
    run = run or time.strftime("%Y%m%d-%H%M%S")
    chosen = SMOKE if smoke or configs == "smoke" else FULL
    if only:
        names = set(only.split(","))
        chosen = [c for c in chosen if c["label"] in names]
    for c in chosen:  # CLI overrides for a cheaper or longer run
        if webnav_steps:
            c["webnav_steps"] = webnav_steps
        if doom_decisions:
            c["doom_decisions"] = doom_decisions
        if wiki_challenges >= 0:
            c["wiki_challenges"] = wiki_challenges
    print(f"run {run}: {[c['label'] for c in chosen]} on {GPU}")
    results = list(evaluate.starmap([(c, GPU, run) for c in chosen]))
    total = sum(r["cost_usd"] for r in results)
    print(f"\n=== run {run} · total ${total:.2f} ===")
    for r in results:
        print(f"\n{r['label']}  ({r['wall_seconds'] / 60:.1f} min, ${r['cost_usd']:.2f})")
        for suite, out in r["suites"].items():
            for agent, m in out.items():
                keep = {k: v for k, v in m.items() if k not in ("per_episode", "races", "agent")}
                print(f"  {suite:9s} {agent:9s} {json.dumps(keep)}")
    print(f"\nmodal volume get s1-data eval/{run} results/")
