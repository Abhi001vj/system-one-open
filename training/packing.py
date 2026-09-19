"""Turn labelled tasks into packed training batches.

One example = one task: the shared prefix followed by every question's branch, with a
tree mask so branches never see each other. Loss is read at each branch's last token,
over that question's allowed answer tokens only — exactly what inference reads.

A batch is several such sequences, right-padded, with a per-row 4D mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from system_one.backends.base import DEFAULT_SYSTEM, BranchSpec
from system_one.decider import Decider
from system_one.types import from_spec


@dataclass
class Example:
    prefix_ids: list[int]
    branches: list[list[int]]
    answer_ids: list[list[list[int]]]  # branch -> answer -> token ids
    targets: list[list[float]]  # branch -> answer -> probability
    family: str
    task_id: str

    @property
    def length(self) -> int:
        return len(self.prefix_ids) + sum(len(b) for b in self.branches)


def build_example(backend, task: dict, system: str = DEFAULT_SYSTEM) -> Example | None:
    """Tokenize one labelled task with the backend that will be trained."""
    questions = [from_spec(q) for q in task["questions"]]
    target = task.get("target", {})
    keep = [q for q in questions if target.get(q.key, {}).get("probs")]
    if not keep:
        return None
    head, tail = backend.split_template(system, Decider.user_head(task["state"], task.get("preamble", "")))
    prefix_ids = backend.tok.encode(head, add_special_tokens=False)
    branches, answer_ids, targets = [], [], []
    for q in keep:
        spec = BranchSpec(*q.render())
        branches.append(backend.tok.encode("\n\n" + spec.body + tail, add_special_tokens=False))
        answer_ids.append([backend._answer_ids(v) for v in spec.answers])
        targets.append(list(target[q.key]["probs"]))
    return Example(prefix_ids, branches, answer_ids, targets, task.get("family", "?"), task.get("id", "?"))


def collate(examples: list[Example], pad_id: int, device: str) -> dict:
    """Right-pad examples into one batch with per-row tree masks."""
    lengths = [e.length for e in examples]
    L = max(lengths)
    B = len(examples)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    mask = torch.zeros((B, 1, L, L), dtype=torch.bool)
    read_rows, read_cols, answer_rows = [], [], []
    for b, ex in enumerate(examples):
        P = len(ex.prefix_ids)
        ids = list(ex.prefix_ids)
        owner = [-1] * P  # -1 marks the shared prefix
        pos = list(range(P))
        for i, branch in enumerate(ex.branches):
            ids += branch
            owner += [i] * len(branch)
            pos += range(P, P + len(branch))
            read_rows.append(b)
            read_cols.append(len(ids) - 1)
            answer_rows.append((b, i))
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids)
        position_ids[b, :n] = torch.tensor(pos)
        own = torch.tensor(owner)
        causal = torch.arange(n)[:, None] >= torch.arange(n)[None, :]
        same = (own[:, None] == own[None, :]) | (own[None, :] == -1)  # everyone sees the prefix
        mask[b, 0, :n, :n] = causal & same
        mask[b, 0, :, 0] = True  # padded rows must attend to something
    return {
        "input_ids": input_ids.to(device),
        "position_ids": position_ids.to(device),
        "attention_mask": mask.to(device),
        "read_rows": torch.tensor(read_rows, device=device),
        "read_cols": torch.tensor(read_cols, device=device),
        "answer_rows": answer_rows,
        "examples": examples,
    }


def kl_loss(logp: torch.Tensor, batch: dict, backend) -> tuple[torch.Tensor, dict]:
    """KL(teacher ‖ student) over each question's allowed answer tokens.

    `logp` is log-softmax over the full vocabulary at every read position.
    """
    device = logp.device
    total, n, agree = logp.new_zeros(()), 0, 0
    for row, (b, i) in enumerate(batch["answer_rows"]):
        ex = batch["examples"][b]
        groups = ex.answer_ids[i]
        scores = torch.stack([torch.logsumexp(logp[row, torch.tensor(g, device=device)], 0) for g in groups])
        student = torch.log_softmax(scores, 0)
        teacher = torch.tensor(ex.targets[i], device=device, dtype=student.dtype).clamp_min(1e-6)
        teacher = teacher / teacher.sum()
        total = total + (teacher * (teacher.log() - student)).sum()
        agree += int(student.argmax().item() == teacher.argmax().item())
        n += 1
    return total / max(n, 1), {"questions": n, "agreement": agree / max(n, 1)}


def soften_targets(probs: list[float], temp: float = 1.0, smooth: float = 0.0) -> list[float]:
    """Flatten a saturated teacher distribution: p^(1/temp) renormalized, then mixed with uniform."""
    p = [max(x, 1e-9) ** (1.0 / temp) for x in probs]
    total = sum(p)
    p = [x / total for x in p]
    if smooth:
        u = 1.0 / len(p)
        p = [(1 - smooth) * x + smooth * u for x in p]
    return p
