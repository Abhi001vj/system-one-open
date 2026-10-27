"""Wikirace eval: the demo's two agents, fixed challenge set, machine-readable results."""

from __future__ import annotations

from rich.console import Console

from system_one.decider import Decider
from system_one.demos.wikirace import AutoregressiveAgent, ParallelAgent, Wiki, race

CHALLENGES = [
    ("Rubber duck", "Roman Empire"),
    ("K-pop", "Knuth–Morris–Pratt algorithm"),
    ("Banana", "Quantum mechanics"),
    ("Taylor Swift", "Photosynthesis"),
    ("Chess", "Volcano"),
    ("Coffee", "Higgs boson"),
    ("Bicycle", "Antarctica"),
    ("Pizza", "Gravitational wave"),
]


def run(decider: Decider, which: str, challenges=None, max_hops: int = 10, links: int = 300,
        shortlist: int = 8, console: Console | None = None) -> dict:
    console = console or Console()
    agent = ParallelAgent(decider, shortlist) if which == "wrapper" else AutoregressiveAgent(decider, links)
    agent.name = which
    wiki = Wiki()
    results = []
    for start, target in (challenges or CHALLENGES):
        console.rule(f"[{which}] {start} → {target}")
        res = race(agent, wiki, start, target, max_hops, links, console)
        res.update(start=start, target=target)
        results.append(res)
    n = len(results)
    finished = [r for r in results if r["finished"]]
    return {
        "agent": which, "challenges": n, "finished": len(finished),
        "finish_rate": round(len(finished) / n, 3),
        "hops_when_finished": round(sum(r["hops"] for r in finished) / max(len(finished), 1), 2),
        "model_seconds": round(sum(r["model_seconds"] for r in results), 1),
        "hallucinations": sum(r["hallucinations"] for r in results),
        "races": results,
    }
