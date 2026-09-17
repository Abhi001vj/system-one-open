"""Turn any open model into a fast, typed, calibrated decision function."""

from .backends import PRESETS, load_backend
from .decider import Decider, Result
from .types import Bool, Choice, Score

__all__ = ["Decider", "Result", "Bool", "Choice", "Score", "PRESETS", "load_backend"]
