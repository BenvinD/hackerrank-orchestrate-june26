"""Evaluation harness: score the system on labeled sample rows and compare models."""

from .metrics import Metrics, score
from .harness import EvalRun, run_model

__all__ = ["Metrics", "score", "EvalRun", "run_model"]
