"""Build the calibration training set: (state, questions) tasks, one JSON per line.

Families are deliberately different in shape so we can hold whole families out:

  doom     real logged game states from runs/ or results/ (goal/fire/target/move/dodge)
  ticket   synthetic support tickets with the 27-question schema from the race demo
  clf      cached classification datasets re-cast as Choice/Bool questions (gold labels)

Augmentations (applied to a copy, linked by `variant_of`):
  shuffle  option order permuted — exposes answer-letter bias
  negate   Bool questions restated in the opposite direction — target must flip

Targets are filled in later by the teacher (`modal_app/label.py`); gold labels, when a
family has them, are written now as `gold`.

    python training/build_dataset.py --out training/data/tasks.jsonl --limit-per-family 400
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re

from system_one.demos.race import QUESTIONS as TICKET_QUESTIONS
from system_one.types import Bool, Choice, Question, Score, to_spec

# ---------------------------------------------------------------- doom

DOOM_PREAMBLE = "You are the tactical brain of a Doom player. Standing orders: {order}"
DOOM_ORDERS = [
    "Survive and kill as many enemies as possible. Pick up health when hurt.",
    "Clear the level aggressively; ignore items unless health is below 40.",
    "Stay alive above all else; retreat from anything stronger than an imp.",
    "Collect weapons and ammo first, then hunt enemies.",
]
DOOM_PARAPHRASE = {
    "fire": [
        "Is an enemy in the crosshair, so the player should fire this instant?",
        "Is a hostile lined up in front of the player right now, making this the moment to shoot?",
        "Would firing the weapon at this exact tick hit an enemy?",
    ],
    "goal": [
        "Considering the player, enemies and items, what is the player's highest-priority goal right now?",
        "What should the player be trying to achieve at this moment?",
        "Given the state above, which objective matters most right now?",
    ],
    "movement": [
        "Given the current situation, how should the player move right now?",
        "Which movement does this moment call for?",
        "How should the player reposition at this tick?",
    ],
    "dodge": [
        "What does this exact moment call for to avoid damage?",
        "Which evasive action best avoids incoming damage right now?",
        "How should the player dodge at this instant?",
    ],
    "target": [
        "Which enemy should the player focus on?",
        "Which hostile deserves the player's attention right now?",
        "Which enemy is the priority target?",
    ],
    "item": [
        "Which item is most worth picking up now?",
        "Which pickup should the player head for?",
        "Which item has the most value to the player right now?",
    ],
}


def doom_tasks(paths: list[str], limit: int, rng: random.Random) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f]
    rng.shuffle(rows)
    seen, tasks = set(), []
    for row in rows:
        state = row["state"]
        if state in seen:  # logs contain long stretches of identical states
            continue
        seen.add(state)
        qs: list[Question] = []
        for key, ans in row["answers"].items():
            texts = DOOM_PARAPHRASE.get(key)
            if not texts:
                continue
            question = rng.choice(texts)
            if ans["type"] == "bool":
                qs.append(Bool(key, question))
            elif ans["type"] == "choice" and len(ans["probabilities"]) > 1:
                qs.append(Choice(key, question, options=list(ans["probabilities"])))
        if len(qs) < 2:
            continue
        tasks.append(
            {
                "family": "doom",
                "state": state,
                "preamble": DOOM_PREAMBLE.format(order=rng.choice(DOOM_ORDERS)),
                "questions": [to_spec(q) for q in qs],
                "meta": {"source": path},
            }
        )
        if len(tasks) >= limit:
            break
    return tasks


# -------------------------------------------------------------- tickets

COMPANIES = ["Northwind Logistics", "Acme Corp", "Brightline Health", "Kestrel Media", "Vantage Robotics", "Solara Bank"]
PLANS = [("Enterprise", "$612k"), ("Enterprise", "$180k"), ("Growth", "$44k"), ("Starter", "$6k")]
ISSUES = [
    ("webhook deliveries return HTTP 500 since your deploy", "integration_failure", "outage"),
    ("the dashboard takes 40+ seconds to load for our whole team", "performance", "degraded"),
    ("two of our admins cannot log in after the SSO change", "access", "degraded"),
    ("we were billed twice for August", "billing_error", "none"),
    ("export jobs silently drop about 5% of rows", "integration_failure", "data_loss"),
    ("we need bulk tagging in the API, which competitors already have", "feature_gap", "none"),
]
TONES = [
    ("Please advise when you can.", "calm"),
    ("This needs a fix today, not a workaround.", "firm"),
    ("If this happens again before renewal we will run an RFP.", "churn_threat"),
    ("Your support has been useless and frankly I'm done explaining this.", "hostile"),
]
EXTRAS = [
    "Our partner launch on Friday depends on this.",
    "We have already restarted the integration twice.",
    "Finance also noticed a duplicate charge this month.",
    "No data appears to have been exposed as far as we can tell.",
    "",
]


def ticket_tasks(limit: int, rng: random.Random) -> list[dict]:
    tasks = []
    for i in range(limit):
        company = rng.choice(COMPANIES)
        plan, arr = rng.choice(PLANS)
        issue, category, impact = rng.choice(ISSUES)
        tone, tone_kind = rng.choice(TONES)
        extra = rng.choice(EXTRAS)
        hours = rng.choice([1, 3, 9, 26])
        state = (
            f"From: {rng.choice(['Priya Raman', 'Tom Alvarez', 'Sofia Lind', 'Dan Okoye'])} "
            f"({rng.choice(['VP Engineering', 'Head of Ops', 'CTO', 'Finance Manager'])})\n"
            f"Account: {company} — {plan} plan, {arr} ARR, renewal in {rng.choice([12, 41, 120, 300])} days\n\n"
            f"For the last {hours} hours, {issue}. {extra} {tone}"
        ).replace("  ", " ")
        qs = rng.sample(TICKET_QUESTIONS, k=rng.randint(6, 12))
        tasks.append(
            {
                "family": "ticket",
                "state": state,
                "preamble": "",
                "questions": [to_spec(q) for q in qs],
                "meta": {"category": category, "impact": impact, "tone": tone_kind, "plan": plan},
            }
        )
    return tasks


# ----------------------------------------------------------- classification


def clf_tasks(limit: int, rng: random.Random) -> list[dict]:
    """Cached datasets as Choice/Bool questions with gold targets."""
    from datasets import load_dataset

    tasks = []
    specs = [
        ("fancyzhx/ag_news", "text", "label", ["World", "Sports", "Business", "Science and technology"],
         "Which section of a newspaper does this story belong to?"),
        ("PolyAI/banking77", "text", "label", None,
         "Which customer-support intent does this message express?"),
    ]
    for name, text_col, label_col, names, question in specs:
        try:
            ds = load_dataset(name, split="train")
        except Exception as exc:  # offline or missing
            print(f"  skip {name}: {exc}")
            continue
        labels = names or ds.features[label_col].names
        idx = rng.sample(range(len(ds)), k=min(limit, len(ds)))
        for i in idx:
            row = ds[i]
            gold = labels[row[label_col]]
            if len(labels) > 12:  # keep cardinality sane: gold + sampled distractors
                options = rng.sample([l for l in labels if l != gold], k=11) + [gold]
            else:
                options = list(labels)
            rng.shuffle(options)
            key = name.split("/")[-1]
            tasks.append(
                {
                    "family": f"clf:{key}",
                    "state": re.sub(r"\s+", " ", row[text_col])[:1200],
                    "preamble": "",
                    "questions": [to_spec(Choice("label", question, options=options))],
                    "gold": {"label": gold},
                    "meta": {"dataset": name},
                }
            )
    return tasks


# ------------------------------------------------------------ augmentation


def augment(task: dict, rng: random.Random) -> list[dict]:
    out = []
    if any(q["type"] == "choice" for q in task["questions"]):
        shuffled = json.loads(json.dumps(task))
        for q in shuffled["questions"]:
            if q["type"] == "choice":
                rng.shuffle(q["options"])
        shuffled["aug"] = "shuffle"
        shuffled["variant_of"] = task["id"]
        out.append(shuffled)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="training/data/tasks.jsonl")
    ap.add_argument("--limit-per-family", type=int, default=400)
    ap.add_argument("--doom-logs", default="results/*/doom-*.jsonl,runs/doom-*.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = [p for pat in args.doom_logs.split(",") for p in sorted(glob.glob(pat))]
    tasks = doom_tasks(paths, args.limit_per_family, rng) if paths else []
    print(f"doom   {len(tasks)} tasks from {len(paths)} logs")
    t = ticket_tasks(args.limit_per_family, rng)
    print(f"ticket {len(t)} tasks")
    tasks += t
    t = clf_tasks(args.limit_per_family // 2, rng)
    print(f"clf    {len(t)} tasks")
    tasks += t

    for i, task in enumerate(tasks):
        task["id"] = f"{task['family']}-{i:06d}"
        task.setdefault("aug", "base")
    if not args.no_augment:
        extra = [v for task in tasks for v in augment(task, rng)]
        for i, v in enumerate(extra):
            v["id"] = f"{v['family']}-aug-{i:06d}"
        tasks += extra
        print(f"aug    {len(extra)} variants")

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    n_q = sum(len(t["questions"]) for t in tasks)
    print(f"wrote {len(tasks)} tasks / {n_q} questions to {args.out}")


if __name__ == "__main__":
    main()
