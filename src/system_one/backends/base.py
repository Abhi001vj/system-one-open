"""Backend contract.

A backend turns (system prompt, shared user head, N question branches) into N
probability distributions over each branch's allowed answers. How it gets
there — tree attention, batched KV tiling, or concurrent 1-token requests — is
the backend's business, and is reported honestly through `exact` and `stats`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

MARK = "⁣Q⁣"  # invisible separator used to split chat templates

DEFAULT_SYSTEM = (
    "You are a fast, calibrated decision function. Read the state, then answer "
    "the question with exactly one of the allowed answers and nothing else."
)


@dataclass
class BranchSpec:
    body: str  # question text appended after the shared head
    answers: list[list[str]]  # per answer: surface-form variants whose probabilities are summed


@dataclass
class ScoreOutput:
    dists: list[list[float]]
    stats: dict = field(default_factory=dict)


class Backend(ABC):
    name: str = "base"
    exact: bool = True  # False when probabilities are reconstructed from top-k logprobs

    @abstractmethod
    def score(self, system: str, user_head: str, branches: list[BranchSpec]) -> ScoreOutput: ...

    @abstractmethod
    def generate_stream(self, system: str, user: str, max_tokens: int = 1024) -> Iterator[str]:
        """Plain autoregressive generation, used as the baseline in demos."""

    def describe(self) -> str:
        return self.name


def group_token_probs(pairs: list[tuple[str, float]], answers: list[list[str]]) -> list[float]:
    """Sum probability of candidate first tokens into answer groups.

    A token belongs to an answer when it is a non-empty prefix of one of that
    answer's variants, or the variant is a prefix of it ("y", "ye", "yes" -> yes).
    Returns unnormalized mass per answer.
    """
    mass = [0.0] * len(answers)
    for token, p in pairs:
        t = token.strip()
        if not t:
            continue
        for i, variants in enumerate(answers):
            if any(v.startswith(t) or t.startswith(v) for v in variants):
                mass[i] += p
                break
    return mass


def normalize(mass: list[float], floor: float = 1e-9) -> list[float]:
    mass = [max(m, floor) for m in mass]
    total = sum(mass)
    return [m / total for m in mass]
