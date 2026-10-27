"""Head-to-head evaluation: one model, two decision mechanisms, three task suites.

    python -m eval.run --model qwen2.5-1.5b --suites webnav,doom,wikirace --out results.json

Every suite runs the wrapper (typed parallel decisions) and the baseline (autoregressive
JSON) on the same weights, so differences come from the mechanism, not model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from rich.console import Console

from system_one.decider import Decider


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--adapter", default="", help="optional LoRA adapter directory to load on top")
    ap.add_argument("--label", default="", help="name for this configuration in the results")
    ap.add_argument("--suites", default="webnav,doom,wikirace")
    ap.add_argument("--agents", default="wrapper,baseline")
    ap.add_argument("--out", default="eval-results.json")
    ap.add_argument("--log-dir", default="")
    # webnav
    ap.add_argument("--webnav-file", default="training/data/mind2web/data/train/train_10.json")
    ap.add_argument("--webnav-steps", type=int, default=100)
    ap.add_argument("--webnav-candidates", type=int, default=30)
    # doom
    ap.add_argument("--doom-seeds", default="1,2,3")
    ap.add_argument("--doom-decisions", type=int, default=200)
    ap.add_argument("--doom-scenario", default="defend_the_center.cfg")
    # wikirace
    ap.add_argument("--wiki-challenges", type=int, default=8)
    ap.add_argument("--wiki-max-hops", type=int, default=10)
    args = ap.parse_args(argv)

    console = Console()
    t0 = time.perf_counter()
    decider = Decider(args.model)
    if args.adapter:
        from peft import PeftModel

        decider.backend.model = PeftModel.from_pretrained(decider.backend.model, args.adapter).eval()
        decider.backend.body = getattr(decider.backend.model.model.model, "language_model", None) or decider.backend.model.model.model
        decider.backend.head = decider.backend.model.get_output_embeddings()
    label = args.label or (args.model + ("+lora" if args.adapter else ""))
    console.print(f"[bold]{label}[/] · {decider.backend.describe()} · loaded in {time.perf_counter() - t0:.0f}s")

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    def logfile(name):
        return open(os.path.join(args.log_dir, f"{name}.jsonl"), "w") if args.log_dir else None

    results = {"label": label, "model": args.model, "adapter": args.adapter,
               "backend": decider.backend.describe(), "suites": {}}
    suites = [s for s in args.suites.split(",") if s]
    agents = [a for a in args.agents.split(",") if a]

    if "webnav" in suites:
        from eval import webnav

        steps = webnav.load_steps(args.webnav_file, args.webnav_steps, args.webnav_candidates)
        console.rule(f"webnav · {len(steps)} steps · {args.webnav_candidates} candidates")
        out = {}
        for which in agents:
            agent = webnav.WrapperAgent(decider) if which == "wrapper" else webnav.BaselineAgent(decider)
            log = logfile(f"webnav-{which}")
            out[which] = webnav.run(agent, steps, log)
            if log:
                log.close()
            console.print(json.dumps(out[which]))
        results["suites"]["webnav"] = out

    if "doom" in suites:
        from eval import doom_suite

        seeds = [int(s) for s in args.doom_seeds.split(",") if s]
        console.rule(f"doom · {args.doom_scenario} · seeds {seeds} · {args.doom_decisions} decisions")
        out = {}
        for which in agents:
            agent = doom_suite.WrapperAgent(decider) if which == "wrapper" else doom_suite.BaselineAgent(decider)
            log = logfile(f"doom-{which}")
            out[which] = doom_suite.run(agent, seeds, args.doom_decisions, log=log, scenario=args.doom_scenario)
            if log:
                log.close()
            console.print(json.dumps({k: v for k, v in out[which].items() if k != "per_episode"}))
        results["suites"]["doom"] = out

    if "wikirace" in suites:
        from eval import wikirace_suite

        console.rule("wikirace")
        out = {}
        for which in agents:
            out[which] = wikirace_suite.run(
                decider, which, wikirace_suite.CHALLENGES[: args.wiki_challenges],
                max_hops=args.wiki_max_hops, console=console,
            )
            console.print(json.dumps({k: v for k, v in out[which].items() if k != "races"}))
        results["suites"]["wikirace"] = out

    results["wall_seconds"] = round(time.perf_counter() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    console.print(f"wrote {args.out} in {results['wall_seconds']:.0f}s")
    return results


if __name__ == "__main__":
    main()
