"""Packed / batched scoring must match one independent forward pass per question."""
import os

import pytest
import torch

from system_one import Bool, Choice, Score
from system_one.backends import DEFAULT_SYSTEM, BranchSpec
from system_one.backends.hf import HFBackend
from system_one.decider import Decider

STATE = (
    "Ticket from Acme Corp (enterprise, $480k ARR): Since last night's deploy our webhook "
    "integration returns HTTP 500 on every call. Orders are not syncing and our partner launch "
    "is Friday. We were also charged twice this month. If this isn't fixed today we will "
    "evaluate other vendors."
)
QUESTIONS = [
    Bool("revenue", "Is revenue currently impacted?"),
    Choice("dept", "Which department should own this?", options=["billing", "technical", "sales", "legal"]),
    Score("churn", "How likely is this customer to churn?", legend=["none", "low", "medium", "high", "certain"]),
    Bool("threat", "Is the language personally threatening?"),
]
MODELS = os.environ.get("S1_TEST_MODELS", "Qwen/Qwen2.5-1.5B-Instruct").split(",")


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("strategy", ["tree", "batch"])
def test_matches_naive(model_id, strategy):
    be = HFBackend(model_id, dtype=torch.float32, strategy="auto")
    if strategy == "tree" and not be.pure_attention:
        pytest.skip("tree attention needs a pure-attention model")
    be.strategy = strategy
    head = Decider.user_head(STATE)
    specs = [BranchSpec(*q.render()) for q in QUESTIONS]
    ref = be.naive_score(DEFAULT_SYSTEM, head, specs)
    for budget in (100_000, 60):  # everything in one pass; one branch per pass
        be.max_pass_tokens = budget
        got = be.score(DEFAULT_SYSTEM, head, specs)
        for q, g, r in zip(QUESTIONS, got.dists, ref):
            assert max(abs(x - y) for x, y in zip(g, r)) < 2e-3, (q.key, g, r)
    again = be.score(DEFAULT_SYSTEM, head, specs)
    if be.pure_attention:
        assert again.stats["reused_tokens"] == again.stats["prefix_tokens"] - 1
