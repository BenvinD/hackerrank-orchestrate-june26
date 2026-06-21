"""The final output row and CSV serialization (exact 14-column contract)."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

from ..enums import OUTPUT_COLUMNS


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _list_or_none(values: list[str]) -> str:
    return ";".join(values) if values else "none"


class OutputRow(BaseModel):
    """One row of output.csv. Field names match the required column names."""

    # echoed inputs
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    # decisions
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: str

    def to_csv_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": _bool(self.evidence_standard_met),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": _list_or_none(self.risk_flags),
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": _list_or_none(self.supporting_image_ids),
            "valid_image": _bool(self.valid_image),
            "severity": self.severity,
        }


def write_output_csv(rows: list[OutputRow], path: str | Path) -> None:
    """Write rows to CSV with the exact required column order, fully quoted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(OUTPUT_COLUMNS), quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())
