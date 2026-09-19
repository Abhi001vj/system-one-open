"""LoRA training against teacher distributions, using the inference layout.

Runs anywhere torch runs: locally on MPS for smoke tests, on a Modal GPU for real runs.
Metrics reported per family: KL, Brier, ECE, teacher agreement.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from system_one.backends.hf import HFBackend

from .packing import build_example, collate, kl_loss, soften_targets


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    labels: str = "training/data/labels.jsonl"
    out_dir: str = "training/out"
    holdout_families: tuple[str, ...] = ("clf:ag_news",)
    val_fraction: float = 0.1
    batch_tokens: int = 4096
    epochs: int = 1
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_steps: int = 0  # 0 = no limit
    # teacher targets come out saturated (0.00/1.00); soften them so the student is not
    # trained to be maximally confident. temp > 1 flattens, smooth mixes in uniform mass.
    target_temp: float = 1.0
    target_smooth: float = 0.0
    seed: int = 0
    log_every: int = 10
    eval_every: int = 0  # 0 = only at the end
    push_to_hub: str = ""
    hub_private: bool = True
    metrics: list[dict] = field(default_factory=list)


def load_examples(backend: HFBackend, path: str, limit: int = 0) -> list:
    out = []
    with open(path) as f:
        for line in f:
            task = json.loads(line)
            ex = build_example(backend, task)
            if ex is not None:
                out.append(ex)
            if limit and len(out) >= limit:
                break
    return out


def batches(examples: list, batch_tokens: int, shuffle: bool, rng: random.Random) -> list[list]:
    order = list(examples)
    if shuffle:
        rng.shuffle(order)
    out, cur, longest = [], [], 0
    for ex in order:
        longest_next = max(longest, ex.length)
        if cur and longest_next * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur, longest = [], 0
            longest_next = ex.length
        cur.append(ex)
        longest = longest_next
    if cur:
        out.append(cur)
    return out


def forward_logp(backend: HFBackend, batch: dict) -> torch.Tensor:
    out = backend.body(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    hidden = out.last_hidden_state[batch["read_rows"], batch["read_cols"]]
    return backend._logp(hidden)


@torch.no_grad()
def evaluate(backend: HFBackend, examples: list, batch_tokens: int, device: str) -> dict:
    """KL / Brier / ECE / agreement, overall and per family."""
    backend.model.eval()
    per_family = defaultdict(lambda: {"kl": 0.0, "brier": 0.0, "n": 0, "agree": 0, "conf": 0.0, "correct": 0.0,
                                      "bins": defaultdict(lambda: [0, 0.0, 0.0])})
    rng = random.Random(0)
    for group in batches(examples, batch_tokens, False, rng):
        batch = collate(group, backend.pad_id, device)
        logp = forward_logp(backend, batch)
        for row, (b, i) in enumerate(batch["answer_rows"]):
            ex = batch["examples"][b]
            groups = ex.answer_ids[i]
            scores = torch.stack([torch.logsumexp(logp[row, torch.tensor(g, device=device)], 0) for g in groups])
            student = torch.softmax(scores.float(), 0)
            teacher = torch.tensor(ex.targets[i], device=device).clamp_min(1e-6)
            teacher = teacher / teacher.sum()
            f = per_family[ex.family]
            f["kl"] += float((teacher * (teacher.log() - student.clamp_min(1e-9).log())).sum())
            f["brier"] += float(((student - teacher) ** 2).sum())
            hit = int(student.argmax() == teacher.argmax())
            f["agree"] += hit
            conf = float(student.max())
            f["conf"] += conf
            f["correct"] += hit
            f["bins"][min(int(conf * 15), 14)][0] += 1
            f["bins"][min(int(conf * 15), 14)][1] += conf
            f["bins"][min(int(conf * 15), 14)][2] += hit
            f["n"] += 1
    result = {}
    for family, f in per_family.items():
        n = max(f["n"], 1)
        ece = sum(cnt / n * abs(c / max(cnt, 1) - acc / max(cnt, 1)) for cnt, c, acc in f["bins"].values())
        result[family] = {
            "n": f["n"], "kl": round(f["kl"] / n, 4), "brier": round(f["brier"] / n, 4),
            "agreement": round(f["agree"] / n, 4), "ece": round(ece, 4), "mean_conf": round(f["conf"] / n, 4),
        }
    tot = {k: 0.0 for k in ("kl", "brier", "agreement", "ece")}
    n_all = sum(v["n"] for v in result.values()) or 1
    for v in result.values():
        for k in tot:
            tot[k] += v[k] * v["n"] / n_all
    result["overall"] = {"n": n_all, **{k: round(v, 4) for k, v in tot.items()}}
    backend.model.train()
    return result


def temperature_baseline(backend: HFBackend, examples: list, batch_tokens: int, device: str) -> dict:
    """Best single temperature on the eval set — the bar a LoRA has to beat."""
    backend.model.eval()
    rows = []
    with torch.no_grad():
        for group in batches(examples, batch_tokens, False, random.Random(0)):
            batch = collate(group, backend.pad_id, device)
            logp = forward_logp(backend, batch)
            for row, (b, i) in enumerate(batch["answer_rows"]):
                ex = batch["examples"][b]
                scores = torch.stack([torch.logsumexp(logp[row, torch.tensor(g, device=device)], 0) for g in ex.answer_ids[i]])
                rows.append((scores.float().cpu(), torch.tensor(ex.targets[i]), ex.family))
    best = None
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]:
        kl = 0.0
        for scores, teacher, _ in rows:
            student = torch.log_softmax(scores / T, 0)
            t = (teacher.clamp_min(1e-6) / teacher.clamp_min(1e-6).sum())
            kl += float((t * (t.log() - student)).sum())
        kl /= max(len(rows), 1)
        if best is None or kl < best[1]:
            best = (T, kl)
    backend.model.train()
    return {"temperature": best[0], "kl": round(best[1], 4), "n": len(rows)}


def train(cfg: TrainConfig) -> dict:
    from peft import LoraConfig, get_peft_model

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    backend = HFBackend(cfg.model, strategy="tree")
    device = backend.device
    if not backend.pure_attention:
        print("warning: model is not pure-attention; tree packing may be invalid for it")

    examples = load_examples(backend, cfg.labels)
    if cfg.target_temp != 1.0 or cfg.target_smooth:
        for ex in examples:
            ex.targets = [soften_targets(t, cfg.target_temp, cfg.target_smooth) for t in ex.targets]
    held = [e for e in examples if e.family in cfg.holdout_families]
    rest = [e for e in examples if e.family not in cfg.holdout_families]
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * cfg.val_fraction))
    val, train_set = rest[:n_val], rest[n_val:]
    print(f"{len(train_set)} train · {len(val)} val · {len(held)} held-out ({', '.join(cfg.holdout_families)})")

    base_eval = evaluate(backend, val + held, cfg.batch_tokens, device)
    base_temp = temperature_baseline(backend, val + held, cfg.batch_tokens, device)
    print("base model:", json.dumps(base_eval["overall"]), "| temperature baseline:", json.dumps(base_temp))

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backend.model = get_peft_model(backend.model, lora)
    backend.body = getattr(backend.model.model.model, "language_model", None) or backend.model.model.model
    backend.head = backend.model.get_output_embeddings()
    backend.model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad], lr=cfg.lr)

    step, t0, history = 0, time.perf_counter(), []
    for epoch in range(cfg.epochs):
        for group in batches(train_set, cfg.batch_tokens, True, rng):
            batch = collate(group, backend.pad_id, device)
            logp = forward_logp(backend, batch)
            loss, info = kl_loss(logp, batch, backend)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in backend.model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0:
                el = time.perf_counter() - t0
                rec = {"step": step, "epoch": epoch, "loss": round(float(loss), 4), "agreement": round(info["agreement"], 3),
                       "examples": len(group), "questions": info["questions"], "s_per_step": round(el / step, 2)}
                history.append(rec)
                print(json.dumps(rec), flush=True)
            if cfg.eval_every and step % cfg.eval_every == 0:
                print("eval:", json.dumps(evaluate(backend, val + held, cfg.batch_tokens, device)), flush=True)
            if cfg.max_steps and step >= cfg.max_steps:
                break
        if cfg.max_steps and step >= cfg.max_steps:
            break

    final = evaluate(backend, val + held, cfg.batch_tokens, device)
    elapsed = time.perf_counter() - t0
    print("final:", json.dumps(final, indent=1), flush=True)

    import os

    os.makedirs(cfg.out_dir, exist_ok=True)
    backend.model.save_pretrained(cfg.out_dir)
    backend.tok.save_pretrained(cfg.out_dir)
    report = {
        "config": {k: v for k, v in cfg.__dict__.items() if k != "metrics"},
        "steps": step, "train_seconds": round(elapsed, 1), "s_per_step": round(elapsed / max(step, 1), 3),
        "base": base_eval, "temperature_baseline": base_temp, "final": final, "history": history,
        "counts": {"train": len(train_set), "val": len(val), "holdout": len(held)},
    }
    with open(os.path.join(cfg.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if cfg.push_to_hub:
        backend.model.push_to_hub(cfg.push_to_hub, private=cfg.hub_private)
        backend.tok.push_to_hub(cfg.push_to_hub, private=cfg.hub_private)
        print("pushed to", cfg.push_to_hub)
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    for name, value in TrainConfig().__dict__.items():
        if name == "metrics":
            continue
        if name == "holdout_families":
            ap.add_argument("--holdout-families", default=",".join(value))
        elif isinstance(value, bool):
            ap.add_argument(f"--{name.replace('_', '-')}", action="store_true" if not value else "store_false")
        else:
            ap.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = ap.parse_args()
    cfg = TrainConfig(**{**vars(args), "holdout_families": tuple(f for f in args.holdout_families.split(",") if f)})
    train(cfg)


if __name__ == "__main__":
    main()
