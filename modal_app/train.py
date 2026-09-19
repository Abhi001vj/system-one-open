"""LoRA calibration training on Modal, with the same packed layout inference uses.

    modal run modal_app/train.py --labels labels-pilot.jsonl --max-steps 40   # cost probe
    modal run modal_app/train.py --labels labels-pilot.jsonl --epochs 1 \
        --push-to-hub <user>/system-one-qwen2.5-1.5b-calib

The report (base vs temperature-scaled baseline vs trained, per family) is written to the
data volume as report-<run>.json and printed at the end.
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
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers>=5.17", "accelerate", "peft", "huggingface_hub", "rich", "requests")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir("src", "/repo/src", copy=True)
    .add_local_dir("training", "/repo/training", copy=True)
    .add_local_file("pyproject.toml", "/repo/pyproject.toml", copy=True)
    .add_local_file("README.md", "/repo/README.md", copy=True)
    .run_commands("pip install --no-deps -e /repo")
)

app = modal.App("system-one-train")
cache = modal.Volume.from_name("s1-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("s1-data", create_if_missing=True)

secrets = [modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])] if os.environ.get("S1_HF_SECRET") else []


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/data": data}, timeout=6 * 60 * 60, secrets=secrets)
def train_remote(labels: str, cfg_overrides: dict, gpu: str, run_name: str) -> dict:
    import sys

    sys.path.insert(0, "/repo")
    from training.trainer import TrainConfig, train

    t0 = time.perf_counter()
    cfg = TrainConfig(labels=f"/data/{labels}", out_dir=f"/data/adapters/{run_name}", **cfg_overrides)
    report = train(cfg)
    report["gpu"] = gpu
    report["wall_seconds"] = round(time.perf_counter() - t0, 1)
    price = GPU_HOURLY.get(gpu, 0)
    report["run_cost_usd"] = round(report["wall_seconds"] / 3600 * price, 3)
    if report["steps"]:
        report["usd_per_1k_steps"] = round(report["s_per_step"] * 1000 / 3600 * price, 2)
    with open(f"/data/report-{run_name}.json", "w") as f:
        json.dump(report, f, indent=1)
    data.commit()
    return report


@app.local_entrypoint()
def main(
    labels: str = "labels-pilot.jsonl",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs: int = 1,
    lr: float = 1e-4,
    batch_tokens: int = 8192,
    max_steps: int = 0,
    lora_r: int = 16,
    holdout_families: str = "clf:ag_news",
    target_temp: float = 1.0,
    target_smooth: float = 0.0,
    push_to_hub: str = "",
    run_name: str = "",
):
    run_name = run_name or f"{model.split('/')[-1]}-{int(time.time())}"
    overrides = dict(
        model=model, epochs=epochs, lr=lr, batch_tokens=batch_tokens, max_steps=max_steps, lora_r=lora_r,
        holdout_families=tuple(f for f in holdout_families.split(",") if f), target_temp=target_temp,
        target_smooth=target_smooth, push_to_hub=push_to_hub,
    )
    report = train_remote.remote(labels, overrides, GPU, run_name)
    print(json.dumps({k: report[k] for k in ("steps", "train_seconds", "s_per_step", "wall_seconds", "run_cost_usd") if k in report}, indent=1))
    print("base      ", json.dumps(report["base"]["overall"]))
    print("temp scale", json.dumps(report["temperature_baseline"]))
    print("trained   ", json.dumps(report["final"]["overall"]))
    for family in report["final"]:
        if family != "overall":
            print(f"  {family:16s} base {json.dumps(report['base'].get(family, {}))} → trained {json.dumps(report['final'][family])}")
    print(f"\nreport: modal volume get s1-data report-{run_name}.json training/out/")
