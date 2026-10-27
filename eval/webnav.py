"""Computer-use eval: pick the right element and operation on real web pages.

Offline and verifiable, from Mind2Web: every step has a goal, the actions taken so far,
the page's candidate elements, and the gold element + operation.

  wrapper   Choice over the candidate elements (tournament when > shortlist) + Choice over
            the operation. The chosen element is always a real candidate.
  baseline  the same candidates in the prompt; the model writes JSON with an element id.
            An id that is not on the page counts as a hallucination.

Both agents see identical candidate lists and use identical weights.
"""

from __future__ import annotations

import json
import random
import re
import time

from system_one.backends import DEFAULT_SYSTEM
from system_one.decider import Decider
from system_one.types import Choice

OPERATIONS = ["CLICK", "TYPE", "SELECT"]
ATTRS = ("aria_label", "aria-label", "title", "placeholder", "value", "alt", "role", "name", "type", "id", "href")


def _attrs(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def describe(node, tag: str, attrs: dict, max_len: int = 110) -> str:
    """One readable line for an element, the way a person would describe it."""
    text = " ".join((node.get_text(" ", strip=True) if node is not None else "").split())[:70]
    bits = []
    for key in ATTRS:
        v = attrs.get(key)
        if v and key not in ("id", "href"):
            bits.append(f'{key.replace("aria_label", "aria-label")}="{str(v)[:40]}"')
        elif v and key == "href":
            bits.append(f'href="{str(v)[:30]}"')
        if len(bits) >= 3:
            break
    label = f'<{tag}>'
    if text:
        label += f' "{text}"'
    if bits:
        label += " " + " ".join(bits)
    return label[:max_len] or f"<{tag}>"


def load_steps(path: str, limit: int, candidates: int, seed: int = 0) -> list[dict]:
    """Flatten Mind2Web episodes into independent steps with rendered candidates."""
    from bs4 import BeautifulSoup

    rng = random.Random(seed)
    with open(path) as f:
        episodes = json.load(f)
    steps = []
    for ep in episodes:
        for i, action in enumerate(ep["actions"]):
            pos = action.get("pos_candidates") or []
            if not pos:
                continue  # no gold element in the candidate set: unscoreable
            op = action["operation"]
            if isinstance(op, str):
                op = json.loads(op.replace("'", '"')) if op.startswith("{") else {"op": op, "value": ""}
            soup = BeautifulSoup(action["cleaned_html"], "html.parser")
            by_id = {}
            for el in soup.find_all(attrs={"backend_node_id": True}):
                by_id[el["backend_node_id"]] = el
            gold_id = pos[0]["backend_node_id"]
            negs = rng.sample(action["neg_candidates"], k=min(candidates - 1, len(action["neg_candidates"])))
            chosen = [pos[0]] + negs
            rng.shuffle(chosen)
            options, gold_index = [], None
            for c in chosen:
                a = _attrs(c["attributes"])
                text = describe(by_id.get(c["backend_node_id"]), c["tag"], a)
                if c["backend_node_id"] == gold_id:
                    gold_index = len(options)
                options.append(text)
            # options must be distinct for a Choice; disambiguate repeats
            seen = {}
            for j, o in enumerate(options):
                if o in seen:
                    options[j] = f"{o} #{j}"
                seen[options[j]] = j
            steps.append(
                {
                    "task": ep["confirmed_task"],
                    "website": ep["website"],
                    "domain": ep["domain"],
                    "history": ep["action_reprs"][:i],
                    "options": options,
                    "gold_index": gold_index,
                    "gold_op": op.get("op", "CLICK"),
                    "gold_value": op.get("value", ""),
                    "annotation_id": ep["annotation_id"],
                    "step": i,
                }
            )
            if len(steps) >= limit:
                return steps
    return steps


def state_text(step: dict) -> str:
    history = "\n".join(f"  {h}" for h in step["history"][-5:]) or "  (none yet)"
    return (
        f"Goal: {step['task']}\nWebsite: {step['website']}\nActions already taken:\n{history}\n"
        f"Elements available on the current page are given as options."
    )


PREAMBLE = (
    "You are operating a web browser to complete the goal. Pick the single element to act on next "
    "and the operation to perform on it."
)
PICK = "Which element should be acted on next to make progress toward the goal?"
OP_Q = "Which operation should be performed on that element?"


class WrapperAgent:
    name = "wrapper"

    def __init__(self, decider: Decider, shortlist: int = 10, group_size: int = 26):
        self.d, self.k, self.group = decider, shortlist, group_size

    def step(self, step: dict) -> dict:
        t0 = time.perf_counter()
        state = state_text(step)
        pool = list(step["options"])
        while len(pool) > self.k:  # tournament rounds, all groups scored in one pass
            groups = [pool[i : i + self.group] for i in range(0, len(pool), self.group)]
            r = self.d.decide(state, [Choice(f"g{i}", PICK, options=g) for i, g in enumerate(groups)], preamble=PREAMBLE)
            pool = []
            for i in range(len(groups)):
                ranked = sorted(r.answers[f"g{i}"]["probabilities"].items(), key=lambda kv: -kv[1])
                pool += [o for o, _ in ranked[:2]]
        r = self.d.decide(
            state,
            [Choice("element", PICK, options=pool), Choice("operation", OP_Q, options=OPERATIONS)],
            preamble=PREAMBLE,
        )
        pick = r.answers["element"]
        return {
            "element": pick["choice"],
            "confidence": pick["confidence"],
            "operation": r.answers["operation"]["choice"],
            "seconds": time.perf_counter() - t0,
            "hallucinated": False,
        }


class BaselineAgent:
    name = "baseline"

    def __init__(self, decider: Decider, max_tokens: int = 120):
        self.be, self.max_tokens = decider.backend, max_tokens

    def step(self, step: dict) -> dict:
        t0 = time.perf_counter()
        listing = "\n".join(f"[{i}] {o}" for i, o in enumerate(step["options"]))
        prompt = (
            f"{PREAMBLE}\n\n{state_text(step)}\n\nElements:\n{listing}\n\n"
            'Reply with ONLY one JSON object: {"element": <index of the element>, "operation": "CLICK" | "TYPE" | "SELECT"}'
        )
        text = "".join(self.be.generate_stream(DEFAULT_SYSTEM, prompt, max_tokens=self.max_tokens))
        m = re.search(r"\{.*?\}", text, re.S)
        element, operation, bad = None, None, True
        if m:
            try:
                obj = json.loads(m.group(0))
                idx = obj.get("element")
                if isinstance(idx, str) and idx.strip().lstrip("[").rstrip("]").isdigit():
                    idx = int(idx.strip().lstrip("[").rstrip("]"))
                if isinstance(idx, int) and 0 <= idx < len(step["options"]):
                    element = step["options"][idx]
                    bad = False
                op = str(obj.get("operation", "")).upper()
                operation = op if op in OPERATIONS else None
            except json.JSONDecodeError:
                pass
        return {
            "element": element,
            "confidence": None,
            "operation": operation,
            "seconds": time.perf_counter() - t0,
            "hallucinated": bad,
            "raw": text[:200],
        }


def run(agent, steps: list[dict], log=None) -> dict:
    correct_el = correct_op = both = halluc = bad_op = 0
    seconds = []
    for i, step in enumerate(steps):
        out = agent.step(step)
        gold_el = step["options"][step["gold_index"]]
        el_ok = out["element"] == gold_el
        op_ok = out["operation"] == step["gold_op"]
        correct_el += el_ok
        correct_op += op_ok
        both += el_ok and op_ok
        halluc += bool(out["hallucinated"])
        bad_op += out["operation"] is None
        seconds.append(out["seconds"])
        if log:
            log.write(json.dumps({"i": i, "agent": agent.name, **{k: v for k, v in out.items() if k != "raw"},
                                  "gold_element": gold_el, "gold_op": step["gold_op"], "element_ok": el_ok,
                                  "op_ok": op_ok, "task": step["task"], "n_options": len(step["options"])}) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  [{agent.name}] {i + 1}/{len(steps)} element {correct_el / (i + 1):.2f} both {both / (i + 1):.2f}", flush=True)
    n = max(len(steps), 1)
    lat = sorted(seconds)
    return {
        "agent": agent.name, "steps": n,
        "element_accuracy": round(correct_el / n, 4),
        "operation_accuracy": round(correct_op / n, 4),
        "step_accuracy": round(both / n, 4),
        "hallucinated_elements": halluc,
        "invalid_operations": bad_op,
        "median_s": round(lat[len(lat) // 2], 3),
        "p90_s": round(lat[int(len(lat) * 0.9)], 3),
        "total_model_s": round(sum(seconds), 1),
    }
