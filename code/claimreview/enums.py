"""Allowed output values, copied verbatim from problem_statement.md.

These are the single source of truth for validation. The rule layer must only
ever emit values present in these sets; anything else collapses to ``unknown``
(or ``none`` where that is the documented fallback).
"""

from __future__ import annotations

CLAIM_OBJECTS: frozenset[str] = frozenset({"car", "laptop", "package"})

CLAIM_STATUSES: frozenset[str] = frozenset(
    {"supported", "contradicted", "not_enough_information"}
)

ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
        "stain",
        "none",
        "unknown",
    }
)

SEVERITIES: frozenset[str] = frozenset({"none", "low", "medium", "high", "unknown"})

RISK_FLAGS: frozenset[str] = frozenset(
    {
        "none",
        "blurry_image",
        "cropped_or_obstructed",
        "low_light_or_glare",
        "wrong_angle",
        "wrong_object",
        "wrong_object_part",
        "damage_not_visible",
        "claim_mismatch",
        "possible_manipulation",
        "non_original_image",
        "text_instruction_present",
        "user_history_risk",
        "manual_review_required",
    }
)

# object_part allowed values are object-specific.
OBJECT_PARTS: dict[str, frozenset[str]] = {
    "car": frozenset(
        {
            "front_bumper",
            "rear_bumper",
            "door",
            "hood",
            "windshield",
            "side_mirror",
            "headlight",
            "taillight",
            "fender",
            "quarter_panel",
            "body",
            "unknown",
        }
    ),
    "laptop": frozenset(
        {
            "screen",
            "keyboard",
            "trackpad",
            "hinge",
            "lid",
            "corner",
            "port",
            "base",
            "body",
            "unknown",
        }
    ),
    "package": frozenset(
        {
            "box",
            "package_corner",
            "package_side",
            "seal",
            "label",
            "contents",
            "item",
            "unknown",
        }
    ),
}

# Exact output column order required by problem_statement.md "Required output".
OUTPUT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
)


def valid_parts_for(claim_object: str) -> frozenset[str]:
    """Allowed ``object_part`` values for a given claim object (empty if unknown object)."""
    return OBJECT_PARTS.get(claim_object, frozenset())


# Union of every object_part across all objects (plus ``unknown``). Used to
# loosely validate model output before the rule layer enforces per-object validity.
ALL_OBJECT_PARTS: frozenset[str] = frozenset().union(*OBJECT_PARTS.values())

# Risk flags that describe image quality / usability, which a VLM can observe
# directly. The remaining flags (user_history_risk, manual_review_required) are
# decided by the rule layer, not the model.
VISION_RISK_FLAGS: frozenset[str] = frozenset(
    {
        "blurry_image",
        "cropped_or_obstructed",
        "low_light_or_glare",
        "wrong_angle",
        "wrong_object",
        "wrong_object_part",
        "damage_not_visible",
        "claim_mismatch",
        "possible_manipulation",
        "non_original_image",
        "text_instruction_present",
    }
)


# Ordering for severity so we can take the "most severe" across images.
SEVERITY_RANK: dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "unknown": -1,
}


def max_severity(values: list[str]) -> str:
    """Most severe known severity in the list; 'unknown' if none are known."""
    known = [v for v in values if SEVERITY_RANK.get(v, -1) >= 0]
    if not known:
        return "unknown"
    return max(known, key=lambda v: SEVERITY_RANK[v])


def coerce(value: object, allowed: frozenset[str], fallback: str) -> str:
    """Return ``value`` if it is an allowed enum string, else ``fallback``."""
    if isinstance(value, str) and value in allowed:
        return value
    return fallback


def coerce_list(values: object, allowed: frozenset[str]) -> list[str]:
    """Keep only allowed, de-duplicated flag strings, preserving first-seen order."""
    if not isinstance(values, (list, tuple, set)):
        return []
    out: list[str] = []
    for v in values:
        if isinstance(v, str) and v in allowed and v not in out:
            out.append(v)
    return out
