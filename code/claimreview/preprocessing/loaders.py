"""CSV loaders for the dataset.

Uses the stdlib ``csv`` module (handles the multi-line, comma-bearing quoted
``user_claim`` cells correctly) so loading is deterministic and dependency-free.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..schema import ClaimInput, EvidenceRequirement, UserHistory


def load_claims(path: str | Path) -> list[ClaimInput]:
    """Load claims.csv / sample_claims.csv. Only input columns are read; any
    expected-output columns present (as in sample_claims.csv) are ignored here."""
    rows: list[ClaimInput] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                ClaimInput(
                    user_id=(row.get("user_id") or "").strip(),
                    image_paths=(row.get("image_paths") or "").strip(),
                    user_claim=(row.get("user_claim") or "").strip(),
                    claim_object=(row.get("claim_object") or "").strip(),
                )
            )
    return rows


def load_user_history(path: str | Path) -> dict[str, UserHistory]:
    """Load user_history.csv into a ``user_id -> UserHistory`` lookup."""
    history: dict[str, UserHistory] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = (row.get("user_id") or "").strip()
            if not uid:
                continue
            history[uid] = UserHistory(
                user_id=uid,
                past_claim_count=row.get("past_claim_count", 0),
                accept_claim=row.get("accept_claim", 0),
                manual_review_claim=row.get("manual_review_claim", 0),
                rejected_claim=row.get("rejected_claim", 0),
                last_90_days_claim_count=row.get("last_90_days_claim_count", 0),
                history_flags=(row.get("history_flags") or "none").strip(),
                history_summary=(row.get("history_summary") or "").strip(),
            )
    return history


def load_evidence_requirements(path: str | Path) -> list[EvidenceRequirement]:
    """Load evidence_requirements.csv."""
    reqs: list[EvidenceRequirement] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            reqs.append(
                EvidenceRequirement(
                    requirement_id=(row.get("requirement_id") or "").strip(),
                    claim_object=(row.get("claim_object") or "").strip(),
                    applies_to=(row.get("applies_to") or "").strip(),
                    minimum_image_evidence=(row.get("minimum_image_evidence") or "").strip(),
                )
            )
    return reqs
