"""Scoring of predictions against the labeled sample rows."""

from __future__ import annotations

from pydantic import BaseModel

from ..enums import SEVERITY_RANK
from ..rules.output import OutputRow

# Columns scored by exact string match.
DECISION_COLS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "valid_image",
    "severity",
]
# Columns scored as sets (semicolon-joined; "none" == empty).
SET_COLS = ["risk_flags", "supporting_image_ids"]


def _set(value: str) -> set[str]:
    return set(filter(None, value.split(";"))) - {"none"}


class Metrics(BaseModel):
    n: int
    column_accuracy: dict[str, float]          # decision + set columns (exact)
    status_confusion: dict[str, int]           # "want->got" -> count
    severity_within_one: float                 # severity off by <=1 level
    risk_precision: float
    risk_recall: float
    risk_f1: float
    manual_review_recall: float                 # caught the manual_review_required rows


def score(predictions: list[OutputRow], expected: list[dict]) -> Metrics:
    n = len(predictions)
    col_correct = {c: 0 for c in DECISION_COLS + SET_COLS}
    status_conf: dict[str, int] = {}
    sev_within = 0
    tp = fp = fn = 0
    mrr_tp = mrr_fn = 0

    for pred, exp in zip(predictions, expected):
        d = pred.to_csv_dict()

        for c in DECISION_COLS:
            if d[c].strip() == (exp.get(c) or "").strip():
                col_correct[c] += 1
        for c in SET_COLS:
            if _set(d[c]) == _set(exp.get(c) or ""):
                col_correct[c] += 1

        gs, ws = SEVERITY_RANK.get(d["severity"], -1), SEVERITY_RANK.get((exp.get("severity") or "").strip(), -1)
        if gs >= 0 and ws >= 0 and abs(gs - ws) <= 1:
            sev_within += 1
        elif d["severity"].strip() == (exp.get("severity") or "").strip():
            sev_within += 1  # both 'unknown'

        key = f"{(exp.get('claim_status') or '').strip()}->{d['claim_status']}"
        status_conf[key] = status_conf.get(key, 0) + 1

        g, w = _set(d["risk_flags"]), _set(exp.get("risk_flags") or "")
        tp += len(g & w)
        fp += len(g - w)
        fn += len(w - g)
        if "manual_review_required" in w:
            mrr_tp += int("manual_review_required" in g)
            mrr_fn += int("manual_review_required" not in g)

    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mrr_rec = mrr_tp / (mrr_tp + mrr_fn) if (mrr_tp + mrr_fn) else 1.0

    return Metrics(
        n=n,
        column_accuracy={c: col_correct[c] / n for c in col_correct} if n else {},
        status_confusion=status_conf,
        severity_within_one=sev_within / n if n else 0.0,
        risk_precision=prec,
        risk_recall=rec,
        risk_f1=f1,
        manual_review_recall=mrr_rec,
    )
