"""Side-by-side race: 27 typed questions about one support ticket.

Left:  parallel typed decisions (one scoring pass).
Right: the same questions as an autoregressive JSON answer, streamed token by token,
       then parsed and validated.

By default both sides use the SAME weights, so the race isolates the decoding
method rather than model quality. `--llm` points the right side elsewhere.

    s1-race --model qwen2.5-1.5b
    s1-race --model llamacpp --llm ollama:gemma4:26b
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..backends import DEFAULT_SYSTEM, load_backend
from ..decider import Decider
from ..types import Bool, Choice, Score

TICKET = """\
From: Priya Raman <priya.raman@northwind-logistics.com> (VP Engineering)
Account: Northwind Logistics — Enterprise plan, $612k ARR, renewal in 41 days, customer for 4 years
Subject: URGENT: webhooks failing since your 02:00 UTC deploy

Since your deploy at 02:00 UTC every webhook delivery to our order service returns HTTP 500 from
your side (request ids attached, e.g. req_8f2c1, req_8f2c7). No shipment events have synced for
9 hours; our warehouse in Rotterdam is picking orders blind and we've already missed two carrier
cutoffs. Our partner launch with Maersk goes live Friday and depends on this integration.
Separately, finance noticed we were billed twice for August. I need a root cause and a fix today,
not a workaround. Honestly, if this happens again before renewal we will run an RFP. Please loop
in whoever owns the platform, not tier-1 support."""

QUESTIONS = [
    Bool("revenue_impacted", "Is the customer's revenue currently impacted?"),
    Choice("business_impact", "What business impact is described?", options=["none", "degraded", "outage", "data_loss"]),
    Bool("integration_issue", "Is an integration issue present?"),
    Choice("account_health", "What is the account health status?", options=["healthy", "watch", "at_risk", "churning"]),
    Choice("incident_scope", "Which incident scope fits best?", options=["single_user", "single_account", "multi_account", "platform_wide"]),
    Bool("security_concern", "Is a security concern present?"),
    Bool("duplicate_charge", "Is a duplicate charge reported?"),
    Score("churn_likelihood", "How likely is this customer to churn?", legend=["very unlikely", "unlikely", "possible", "likely", "very likely"]),
    Score("scope_certainty", "How certain is the scope of the incident?", legend=["unknown", "vague", "partly clear", "clear"]),
    Bool("human_attention", "Is human attention needed?"),
    Score("security_risk", "What is the security risk level?", legend=["none", "low", "medium", "high"]),
    Bool("feature_request", "Is this primarily a feature request?"),
    Bool("partner_launch_endangered", "Is a partner launch endangered?"),
    Bool("credible_churn_risk", "Is there a credible churn risk?"),
    Bool("production_down", "Is a production capability down?"),
    Bool("customer_data_exposed", "Has customer data been exposed?"),
    Bool("server_error_reported", "Is a server error reported?"),
    Score("financial_impact", "What is the financial impact level?", legend=["none", "minor", "moderate", "major"]),
    Choice("response_deadline", "Which response deadline applies?", options=["within_hour", "today", "this_week", "no_deadline"]),
    Bool("repeated_failures", "Does the customer mention repeated production failures?"),
    Choice("primary_department", "Which primary department should own this?", options=["billing", "technical", "sales", "legal", "customer_success"]),
    Choice("requested_resolution", "Which resolution is requested?", options=["refund", "restore_service", "root_cause", "account_change", "information"]),
    Bool("threatening_language", "Is the language personally threatening?"),
    Bool("concrete_deadline", "Is a concrete deadline stated?"),
    Choice("issue_category", "Which issue category fits best?", options=["billing_error", "integration_failure", "performance", "access", "feature_gap"]),
    Score("technical_specificity", "How technically specific is the report?", legend=["none", "low", "medium", "high"]),
    Score("resolution_complexity", "How complex will resolution be?", legend=["trivial", "simple", "moderate", "complex"]),
]


def json_prompt(state: str) -> str:
    lines = ["STATE:", state, "", "Answer every question below. Return ONLY one JSON object with exactly these keys.", ""]
    for q in QUESTIONS:
        if isinstance(q, Bool):
            spec = "true or false"
        elif isinstance(q, Choice):
            spec = "one of " + ", ".join(f'"{o}"' for o in q.options)
        else:
            spec = f"integer 0-{len(q.legend) - 1} ({'; '.join(f'{i}={d}' for i, d in enumerate(q.legend))})"
        lines.append(f'"{q.key}": {q.question} -> {spec}')
    return "\n".join(lines)


def validate(text: str) -> tuple[dict | None, list[str]]:
    """Parse the LLM's JSON and list every schema violation."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, ["no JSON object found"]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e.msg}"]
    errors = []
    for q in QUESTIONS:
        if q.key not in obj:
            errors.append(f"missing {q.key}")
            continue
        v = obj[q.key]
        ok = (
            isinstance(v, bool) if isinstance(q, Bool)
            else v in q.options if isinstance(q, Choice)
            else isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(q.legend)
        )
        if not ok:
            errors.append(f"{q.key}: invalid value {v!r}")
    extra = set(obj) - {q.key for q in QUESTIONS}
    errors += [f"hallucinated key {k}" for k in sorted(extra)]
    return obj, errors


def parallel_value(a: dict):
    return a["value"] if a["type"] == "bool" else a["choice"] if a["type"] == "choice" else a["level"]


def fmt_answer(a: dict) -> Text:
    if a["type"] == "bool":
        p = a["p_true"]
        return Text.assemble(("true " if a["value"] else "false", "bold green" if a["value"] else "bold red"), f"  p={p:.2f}")
    if a["type"] == "choice":
        return Text.assemble((a["choice"], "bold cyan"), f"  conf={a['confidence']:.2f}")
    return Text.assemble((f"{a['score']:.2f}", "bold yellow"), f"  lvl={a['level']} conf={a['confidence']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-1.5b", help="backend spec or preset for the parallel side")
    ap.add_argument("--llm", default=None, help="backend spec for the autoregressive side (default: same weights)")
    ap.add_argument("--concurrent", action="store_true", help="start both sides at once (they share the GPU)")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--save", default=None, help="write a JSON record of the run")
    args = ap.parse_args()

    console = Console()
    with console.status(f"loading {args.model}…"):
        decider = Decider(args.model)
        llm = decider.backend if args.llm is None else load_backend(args.llm)
        decider.decide(TICKET[:200], QUESTIONS[:2])  # warm kernels, not the prefix
    console.print(f"parallel: [bold]{decider.backend.describe()}[/]   autoregressive: [bold]{llm.describe()}[/]")

    state = {"par": None, "par_t": None, "ar_text": "", "ar_t": None, "ar_first": None, "t0": None}

    def run_parallel():
        r = decider.decide(TICKET, QUESTIONS)
        state["par"], state["par_t"] = r, time.perf_counter() - state["t0"]

    def run_ar():
        start = time.perf_counter()
        for piece in llm.generate_stream(DEFAULT_SYSTEM, json_prompt(TICKET), max_tokens=args.max_tokens):
            if state["ar_first"] is None:
                state["ar_first"] = time.perf_counter() - start
            state["ar_text"] += piece
        state["ar_t"] = time.perf_counter() - state["t0"] if args.concurrent else time.perf_counter() - start

    def render() -> Layout:
        now = time.perf_counter() - state["t0"]
        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(style="dim", no_wrap=True)
        tbl.add_column(no_wrap=True)
        if state["par"]:
            for q in QUESTIONS:
                tbl.add_row(q.key, fmt_answer(state["par"].answers[q.key]))
        left_footer = (
            Text(f"completed in {state['par_t']:.3f}s  ·  {len(QUESTIONS)} typed answers  ·  0 type errors", style="bold green")
            if state["par_t"] else Text(f"scoring… {now:.2f}s", style="yellow")
        )
        right_text = Text(state["ar_text"][-3000:] or "waiting for first token…", overflow="fold")
        if state["ar_t"]:
            _, errs = validate(state["ar_text"])
            right_footer = Text(
                f"completed in {state['ar_t']:.3f}s  ·  {len(errs)} schema errors", style="bold red" if errs else "bold green"
            )
        else:
            right_footer = Text(f"generating… {len(state['ar_text'])} chars", style="yellow")
        lay = Layout()
        lay.split_row(
            Layout(Panel(Group(tbl, Text(""), left_footer), title="parallel typed decisions", border_style="green")),
            Layout(Panel(Group(right_text, Text(""), right_footer), title="autoregressive JSON", border_style="magenta")),
        )
        return lay

    state["t0"] = time.perf_counter()
    with Live(render(), console=console, refresh_per_second=20, screen=False) as live:
        if args.concurrent:
            threads = [threading.Thread(target=f, daemon=True) for f in (run_parallel, run_ar)]
            for t in threads:
                t.start()
            while any(t.is_alive() for t in threads):
                live.update(render())
                time.sleep(0.05)
        else:
            run_parallel()
            live.update(render())
            state["t0"] = time.perf_counter()
            th = threading.Thread(target=run_ar, daemon=True)
            th.start()
            while th.is_alive():
                live.update(render())
                time.sleep(0.05)
        live.update(render())

    obj, errs = validate(state["ar_text"])
    agree = total = 0
    if obj:
        for q in QUESTIONS:
            if q.key in obj:
                total += 1
                agree += obj[q.key] == parallel_value(state["par"].answers[q.key])
    speedup = state["ar_t"] / state["par_t"]
    console.print(
        f"\n[bold]parallel[/] {state['par_t']*1000:.0f} ms   [bold]autoregressive[/] {state['ar_t']*1000:.0f} ms "
        f"(first token {state['ar_first']*1000:.0f} ms)   → [bold green]{speedup:.1f}× faster[/]"
    )
    console.print(f"agreement on parsed keys: {agree}/{total}   schema errors on autoregressive side: {len(errs)}")
    for e in errs[:10]:
        console.print(f"  [red]•[/] {e}")
    console.print(f"[dim]parallel stats: {state['par'].stats}[/]")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(
                {
                    "parallel_backend": decider.backend.describe(), "llm_backend": llm.describe(),
                    "parallel_s": state["par_t"], "llm_s": state["ar_t"], "llm_first_token_s": state["ar_first"],
                    "agreement": [agree, total], "schema_errors": errs, "parallel": state["par"].answers,
                    "llm_raw": state["ar_text"], "concurrent": args.concurrent, "stats": state["par"].stats,
                },
                f, indent=1,
            )


if __name__ == "__main__":
    main()
