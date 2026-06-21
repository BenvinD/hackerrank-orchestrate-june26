"""Layer 1: pure-Python preprocessing (no model calls).

Loads the dataset CSVs, joins user history, attaches the evidence requirements
relevant to each claim object, and encodes (downscales) the submitted images.
"""

from .loaders import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from .images import encode_image
from .pipeline import prepare_claims, prepare_dataset

__all__ = [
    "load_claims",
    "load_evidence_requirements",
    "load_user_history",
    "encode_image",
    "prepare_claims",
    "prepare_dataset",
]
