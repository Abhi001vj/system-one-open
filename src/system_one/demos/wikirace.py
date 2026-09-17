"""Wikirace: reach a target Wikipedia page using only links on the current page.

Parallel agent, per hop (default --stage1 groups):
  round    links split into groups of <=26 lettered options, one Choice per group,
           all groups scored in one parallel pass; the top 2 of each group advance
  final    once <= shortlist remain, one Choice picks the link
  (--stage1 bool scores one yes/no question per link instead, then a Choice over the top-k)
The chosen value is always a real link — hallucination is impossible by construction.

Autoregressive baseline (--baseline): the same model sees the same link list and
must type the title of the link to follow. Invalid titles are counted as
hallucinations and the hop is retried (up to 3 times, then a random link).

    s1-wikirace --model qwen2.5-1.5b --start "Rubber duck" --target "Roman Empire"
    s1-wikirace --model llamacpp --baseline
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field

import requests
from rich.console import Console
from rich.table import Table

from ..backends import DEFAULT_SYSTEM
from ..decider import Decider
from ..types import Bool, Choice

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "system-one-open-wikirace/0.1 (research demo; https://github.com/Abhi001vj/system-one-open)"}
SKIP = re.compile(r"^(List of|Index of|Outline of)|\(disambiguation\)|^\d{3,4}$|^\d{1,2} \w+$|identifier|ISBN|Wayback")

END_SECTIONS = ["See_also", "Notes", "References", "Citations", "Sources", "Further_reading", "External_links", "Bibliography"]

CHALLENGES = [
    ("Rubber duck", "Roman Empire"),
    ("K-pop", "Knuth–Morris–Pratt algorithm"),
    ("Banana", "Quantum mechanics"),
    ("Taylor Swift", "Photosynthesis"),
]


class Wiki:
    def __init__(self):
        self.http = requests.Session()
        self.http.headers.update(HEADERS)
        self._links: dict[str, tuple[str, list[str]]] = {}

    def links(self, title: str, limit: int) -> tuple[str, list[str]]:
        """Resolve redirects and return (canonical title, article links in page order)."""
        if title not in self._links:
            r = self.http.get(
                API,
                params={"action": "parse", "page": title, "prop": "text", "redirects": 1, "format": "json", "formatversion": 2},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()["parse"]
            html = data["text"]
            # body only: drop references, see-also, external links and the navboxes after them
            cut = min([i for i in (html.find(f'id="{h}"') for h in END_SECTIONS) if i > 0], default=len(html))
            html = re.sub(r'<sup[^>]*class="reference".*?</sup>', "", html[:cut], flags=re.S)
            seen, out = set(), []
            for m in re.finditer(r'<a href="/wiki/([^"#?]+)"[^>]*title="([^"]+)"', html):
                path, name = m.group(1), m.group(2).replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
                if ":" in path or name in seen or SKIP.search(name):
                    continue
                seen.add(name)
                out.append(name)
            self._links[title] = (data["title"], out)
        canon, out = self._links[title]
        return canon, out[:limit]

    def summary(self, title: str) -> str:
        r = self.http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}", timeout=30)
        return r.json().get("extract", "") if r.ok else ""


@dataclass
class Hop:
    page: str
    chosen: str
    seconds: float
    candidates: int
    top: list[tuple[str, float]] = field(default_factory=list)
    hallucinations: int = 0


def norm(t: str) -> str:
    return t.replace("_", " ").strip().lower()


PRE = "You are playing Wikirace: reach the target Wikipedia page by clicking links. Fewer clicks is better."
PICK = "Which link should be clicked next to reach the target page fastest?"


class ParallelAgent:
    name = "parallel"

    def __init__(self, decider: Decider, shortlist: int, stage1: str = "groups", group_size: int = 26, advance: int = 2):
        self.d, self.k, self.stage1 = decider, shortlist, stage1
        self.group_size, self.advance = group_size, advance

    def _bool_round(self, state: str, cands: list[str]) -> list[tuple[str, float]]:
        # the instruction lives in the shared prefix so each branch is just the link title
        pre = PRE + " For the link given as the question, answer yes if following it likely gets closer to the target page."
        r = self.d.decide(state, [Bool(f"l{i}", f'Link "{l}"') for i, l in enumerate(cands)], preamble=pre)
        return sorted(((l, r.answers[f"l{i}"]["p_true"]) for i, l in enumerate(cands)), key=lambda x: -x[1])

    def _group_round(self, state: str, cands: list[str]) -> list[tuple[str, float]]:
        groups = [cands[i : i + self.group_size] for i in range(0, len(cands), self.group_size)]
        r = self.d.decide(state, [Choice(f"g{i}", PICK, options=g) for i, g in enumerate(groups)], preamble=PRE)
        survivors = []
        for i in range(len(groups)):
            probs = sorted(r.answers[f"g{i}"]["probabilities"].items(), key=lambda kv: -kv[1])
            survivors += probs[: self.advance]
        return sorted(survivors, key=lambda x: -x[1])

    def step(self, page: str, links: list[str], target: str, target_desc: str, visited: set[str]) -> Hop:
        t0 = time.perf_counter()
        cands = [l for l in links if norm(l) not in visited]
        state = f"Target page: {target}\nAbout the target: {target_desc}\nCurrent page: {page}"
        pool = cands
        ranked: list[tuple[str, float]] = []
        if self.stage1 == "bool" and len(pool) > self.k:
            ranked = self._bool_round(state, pool)
            pool = [l for l, _ in ranked[: self.k]]
        while len(pool) > self.k:
            ranked = self._group_round(state, pool)
            pool = [l for l, _ in ranked]
        final = self.d.decide(state, [Choice("pick", PICK, options=pool)], preamble=PRE).answers["pick"]
        top = sorted(final["probabilities"].items(), key=lambda kv: -kv[1])
        return Hop(page, final["choice"], time.perf_counter() - t0, len(cands), top[:5])


class AutoregressiveAgent:
    name = "autoregressive"

    def __init__(self, decider: Decider, max_links: int):
        self.be, self.max_links = decider.backend, max_links

    def step(self, page: str, links: list[str], target: str, target_desc: str, visited: set[str]) -> Hop:
        t0 = time.perf_counter()
        cands = [l for l in links if norm(l) not in visited][: self.max_links]
        valid = {norm(l): l for l in cands}
        prompt = (
            f"You are playing Wikirace. Target page: {target}\nAbout the target: {target_desc}\nCurrent page: {page}\n\n"
            "Links on the current page:\n" + "\n".join(cands) +
            "\n\nReply with ONLY the exact title of the single link to click next, nothing else."
        )
        halluc = 0
        for _ in range(3):
            text = "".join(self.be.generate_stream(DEFAULT_SYSTEM, prompt, max_tokens=40)).strip().strip('"').splitlines()
            guess = norm(text[0]) if text else ""
            if guess in valid:
                return Hop(page, valid[guess], time.perf_counter() - t0, len(cands), hallucinations=halluc)
            halluc += 1
        return Hop(page, random.choice(cands), time.perf_counter() - t0, len(cands), hallucinations=halluc)


def race(agent, wiki: Wiki, start: str, target: str, max_hops: int, link_limit: int, console: Console) -> dict:
    target_canon, _ = wiki.links(target, 1)
    desc = wiki.summary(target_canon)[:400]
    page, visited, hops = start, set(), []
    for _ in range(max_hops):
        canon, links = wiki.links(page, link_limit)
        visited.add(norm(canon))
        if norm(canon) == norm(target_canon):
            break
        direct = next((l for l in links if norm(l) == norm(target_canon)), None)
        if direct:  # code handles the trivial case
            hops.append(Hop(canon, direct, 0.0, len(links)))
            console.print(f"  [{agent.name}] {canon} → [bold green]{direct}[/] (direct link)")
            page = direct
            continue
        hop = agent.step(canon, links, target_canon, desc, visited)
        hops.append(hop)
        top = ", ".join(f"{l} {p:.2f}" for l, p in hop.top[:3])
        console.print(
            f"  [{agent.name}] {canon} → [bold]{hop.chosen}[/]  {hop.seconds * 1000:.0f} ms over {hop.candidates} links"
            + (f"  [red]{hop.hallucinations} hallucinated[/]" if hop.hallucinations else "")
            + (f"  [dim]top: {top}[/]" if top else "")
        )
        page = hop.chosen
        visited.add(norm(hop.chosen))  # link titles can be redirects to pages already seen
    finished = norm(wiki.links(page, 1)[0]) == norm(target_canon)
    return {
        "agent": agent.name, "finished": finished, "hops": len(hops),
        "model_seconds": sum(h.seconds for h in hops), "hallucinations": sum(h.hallucinations for h in hops),
        "path": [start] + [h.chosen for h in hops],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--start")
    ap.add_argument("--target")
    ap.add_argument("--all", action="store_true", help="run every built-in challenge")
    ap.add_argument("--baseline", action="store_true", help="also run the autoregressive agent")
    ap.add_argument("--links", type=int, default=300, help="max links considered per page")
    ap.add_argument("--baseline-links", type=int, default=300)
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--stage1", choices=["groups", "bool"], default="groups")
    ap.add_argument("--max-hops", type=int, default=12)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    console = Console()
    with console.status(f"loading {args.model}…"):
        decider = Decider(args.model)
    wiki = Wiki()
    challenges = CHALLENGES if args.all else [(args.start or CHALLENGES[0][0], args.target or CHALLENGES[0][1])]
    agents = [ParallelAgent(decider, args.shortlist, args.stage1)] + ([AutoregressiveAgent(decider, args.baseline_links)] if args.baseline else [])
    results = []
    for start, target in challenges:
        console.rule(f"{start} → {target}   ·   {decider.backend.describe()}")
        for agent in agents:
            res = race(agent, wiki, start, target, args.max_hops, args.links, console)
            res.update(start=start, target=target, backend=decider.backend.describe())
            results.append(res)

    tbl = Table(title="Wikirace results")
    for col in ["challenge", "agent", "finished", "hops", "model time", "hallucinations"]:
        tbl.add_column(col)
    for r in results:
        tbl.add_row(
            f"{r['start']} → {r['target']}", r["agent"], "✓" if r["finished"] else "✗", str(r["hops"]),
            f"{r['model_seconds']:.2f}s", str(r["hallucinations"]),
        )
    console.print(tbl)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
