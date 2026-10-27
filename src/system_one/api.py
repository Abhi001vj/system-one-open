"""Jev-compatible `POST /v1/systemone` server on top of any backend.

Speaking the same wire format as Jev (and the other open implementations, e.g.
mmastrac/djev-spark) means Jev clients work against this repo unchanged, and one
evaluation harness can drive every implementation through the same interface.

    pip install -e ".[server]"
    s1-serve --model qwen2.5-1.5b --port 8110

    curl -s localhost:8110/v1/systemone -H 'content-type: application/json' -d '{
      "state": {"ticket": "Everything is down and we have a demo at noon."},
      "questions": {
        "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},
        "team":   {"type": "choice", "instructions": "Which team owns this?",
                   "criteria": {"billing": null, "outage": "service down", "feature": null}},
        "tone":   {"type": "score", "instructions": "How angry is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]}}}'

Question types follow Jev: `noul` (yes/no probability), `choice` (criteria maps option
name to description or null), `score` (criteria is an ordered list of level names).
"""

from __future__ import annotations

import json
import time
from typing import Any

from .decider import Decider
from .types import Bool, Choice, Question, Score


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, indent=1, ensure_ascii=False)


def parse_questions(questions: dict[str, dict]) -> list[Question]:
    out: list[Question] = []
    for key, q in questions.items():
        kind = q.get("type")
        instructions = q.get("instructions") or ""
        criteria = q.get("criteria")
        if kind in ("noul", "boolean", "bool"):
            text = instructions
            if isinstance(criteria, dict) and (criteria.get("true") or criteria.get("false")):
                text += f" (yes means: {criteria.get('true') or 'true'}; no means: {criteria.get('false') or 'false'})"
            out.append(Bool(key, text))
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(f"question {key!r}: choice needs a criteria object")
            options = list(criteria)
            described = [f"{name} — {criteria[name]}" if criteria[name] else name for name in options]
            question = Choice(key, instructions, options=described)
            question._public = options  # map back to the caller's option names
            out.append(question)
        elif kind == "score":
            if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
                raise ValueError(f"question {key!r}: score needs at least two ordered levels")
            out.append(Score(key, instructions, legend=[str(c) for c in criteria]))
        else:
            raise ValueError(f"question {key!r}: unknown type {kind!r}")
    return out


def to_jev_answers(questions: list[Question], answers: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for q in questions:
        a = answers[q.key]
        if a["type"] == "bool":
            out[q.key] = {"noul": round(a["p_true"], 6)}
        elif a["type"] == "choice":
            public = getattr(q, "_public", q.options)
            rename = dict(zip(q.options, public))
            probs = {rename[k]: round(v, 6) for k, v in a["probabilities"].items()}
            out[q.key] = {"choice": rename[a["choice"]], "probabilities": probs,
                          "confidence": round(a["confidence"], 6)}
        else:
            out[q.key] = {"score": round(a["score"], 6), "legend": a["legend"],
                          "probabilities": {str(i): round(p, 6) for i, p in enumerate(a.get("probabilities", []))},
                          "confidence": round(a["confidence"], 6)}
    return out


def create_app(decider: Decider, model_name: str = "system-one-open"):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="system-one-open", description="Jev-compatible structured decisions")

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_name, "backend": decider.backend.describe()}

    @app.post("/v1/systemone")
    def systemone(body: dict):
        if not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise HTTPException(422, {"error": {"message": "questions must be a non-empty object"}})
        try:
            questions = parse_questions(body["questions"])
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": {"message": str(exc)}})
        t0 = time.perf_counter()
        result = decider.decide(render_state(body.get("state", "")), questions,
                                preamble=body.get("instructions", "") or "")
        return {
            "answers": to_jev_answers(questions, result.answers),
            "usage": {"input_tokens": result.stats.get("prefix_tokens"),
                      "output_tokens": result.stats.get("branch_tokens")},
            "diagnostics": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                            "backend": decider.backend.describe(), "exact": decider.backend.exact,
                            **result.stats},
            "model": model_name,
        }

    return app


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-1.5b", help="backend spec or preset")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--name", default="", help="model name reported in responses")
    args = ap.parse_args()
    decider = Decider(args.model)
    uvicorn.run(create_app(decider, args.name or args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
