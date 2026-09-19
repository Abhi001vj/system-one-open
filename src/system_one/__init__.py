"""Turn any open model into a fast, typed, calibrated decision function."""

from .backends import PRESETS, load_backend
from .decider import Decider, Result
from .types import Bool, Choice, Score, answer_labels, from_spec, to_spec

__all__ = [
    "Decider", "Result", "Bool", "Choice", "Score", "PRESETS", "load_backend",
    "to_spec", "from_spec", "answer_labels",
]
