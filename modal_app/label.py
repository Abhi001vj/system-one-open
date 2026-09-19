"""Teacher labelling on Modal: score every task's questions with a big open model.

The teacher is read with the same parallel scorer as inference, so its targets are
exact probability distributions over each question's allowed answers — no sampling,
one forward pass per task.

    modal volume put s1-data training/data/tasks-pilot.jsonl tasks-pilot.jsonl
    modal run modal_app/label.py --tasks tasks-pilot.jsonl --limit 50      # smoke + cost estimate
    modal run modal_app/label.py --tasks tasks-pilot.jsonl                 # full pass
    modal volume get s1-data labels-tasks-pilot.jsonl training/data/

Cost: H100 is $3.95/h (Sep 2026). The first run also pays for the model download into
the cache volume; later runs reuse it.
"""

from __future__ import annotations

import json
import os
import time

import modal

GPU = os.environ.get("S1_GPU", "H100")
TEACHER = os.environ.get("S1_TEACHER", "google/gemma-4-31B-it")
GPU_HOURLY = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
              "H100": 3.95, "H200": 4.54, "B200": 6.25, "B300": 7.10}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers>=5.17", "accelerate", "huggingface_hub[hf_transfer]", "rich", "requests")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("src", "/repo/src", copy=True)
    .add_local_file("pyproject.toml", "/repo/pyproject.toml", copy=True)
    .add_local_file("README.md", "/repo/README.md", copy=True)
    .run_commands("pip install --no-deps -e /repo")
)

app = modal.App("system-one-label")
cache = modal.Volume.from_name("s1-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("s1-data", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache, "/data": data},
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])] if os.environ.get("S1_HF_SECRET") else [],
)
def label(tasks_file: str, teacher: str, limit: int | None, shard: int, shards: int, out_file: str, gpu: str = GPU) -> dict:
    import torch

    from system_one import Decider, answer_labels, from_spec

    t_start = time.perf_counter()
    with open(f"/data/{tasks_file}") as f:
        tasks = [json.loads(line) for line in f]
    tasks = [t for i, t in enumerate(tasks) if i % shards == shard]
    if limit:
        tasks = tasks[:limit]

    decider = Decider(f"hf:{teacher}", dtype=torch.bfloat16, device_map="auto")
    backend = decider.backend
    print(f"loaded {backend.describe()} in {time.perf_counter() - t_start:.0f}s; {len(tasks)} tasks", flush=True)

    t_label = time.perf_counter()
    done, questions, tokens = 0, 0, 0
    path = f"/data/{out_file}" + (f".{shard}" if shards > 1 else "")
    with open(path, "w") as out:
        for task in tasks:
            qs = [from_spec(q) for q in task["questions"]]
            r = decider.decide(task["state"], qs, preamble=task.get("preamble", ""))
            targets = {}
            for q in qs:
                a = r.answers[q.key]
                labels = answer_labels(q)
                if a["type"] == "bool":
                    probs = [a["p_true"], 1 - a["p_true"]]
                elif a["type"] == "choice":
                    probs = [a["probabilities"][o] for o in labels]
                else:
                    probs = None  # score questions carry their own expectation; fill if used
                targets[q.key] = {"labels": labels, "probs": probs}
            out.write(json.dumps({**task, "teacher": teacher, "target": targets, "teacher_stats": r.stats}) + "\n")
            done += 1
            questions += len(qs)
            tokens += r.stats.get("prefix_tokens", 0) + r.stats.get("branch_tokens", 0)
            if done % 50 == 0:
                el = time.perf_counter() - t_label
                print(f"{done}/{len(tasks)} tasks  {questions} questions  {el:.0f}s  {done / el:.2f} tasks/s", flush=True)
    data.commit()

    elapsed = time.perf_counter() - t_label
    total = time.perf_counter() - t_start
    rate = done / elapsed if elapsed else 0
    price = GPU_HOURLY.get(gpu, 0)
    stats = {
        "teacher": teacher, "gpu": gpu, "tasks": done, "questions": questions, "tokens": tokens,
        "load_s": round(t_label - t_start, 1), "label_s": round(elapsed, 1), "total_s": round(total, 1),
        "tasks_per_s": round(rate, 3), "questions_per_s": round(questions / elapsed, 2) if elapsed else 0,
        "gpu_hourly_usd": price, "run_cost_usd": round(total / 3600 * price, 3),
        "usd_per_1k_tasks": round(1000 / rate / 3600 * price, 3) if rate else None,
        "out": path,
    }
    print(json.dumps(stats, indent=1), flush=True)
    return stats


@app.local_entrypoint()
def main(tasks: str = "tasks-pilot.jsonl", teacher: str = TEACHER, limit: int = 0, shards: int = 1, out: str = ""):
    out = out or f"labels-{tasks}"
    args = [(tasks, teacher, limit or None, i, shards, out, GPU) for i in range(shards)]
    results = list(label.starmap(args))
    total_tasks = sum(r["tasks"] for r in results)
    cost = sum(r["run_cost_usd"] for r in results)
    per_1k = [r["usd_per_1k_tasks"] for r in results if r["usd_per_1k_tasks"]]
    print(f"\nlabelled {total_tasks} tasks, run cost ${cost:.2f}")
    if per_1k:
        print(f"steady-state ≈ ${sum(per_1k) / len(per_1k) / shards:.3f} per 1k tasks on {GPU} (excludes model load)")
