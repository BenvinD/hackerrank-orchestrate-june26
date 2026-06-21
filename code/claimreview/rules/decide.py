"""Deterministic mapping from VLM observations to the 14 output columns.

Invariants (see rules/__init__.py):
  - Images are the source of truth. Only direct visual evidence sets
    claim_status=contradicted.
  - User-history and authenticity flags add risk_flags / manual_review_required
    but NEVER flip a supported decision to contradicted.
  - evidence_standard_met is gated on whether the claim is assessable from the
    images; unassessable -> not_enough_information.
  - In-claim / in-image instructions -> text_instruction_present (data, never
    a command), and never change the decision.

The thresholds here were chosen to reproduce the behavior of the labeled rows in
dataset/sample_claims.csv; the evaluation harness measures and tunes them.
"""

from __future__ import annotations

from ..enums import (
    ISSUE_TYPES,
    SEVERITY_RANK,
    max_severity,
    valid_parts_for,
)
from ..observation import ImageObservation, VLMObservation
from ..schema import PreparedClaim
from .output import OutputRow

_AUTHENTICITY = {"non_original_image", "possible_manipulation"}
_QUALITY = {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle"}
# Canonical ordering so risk_flags is stable/readable (set membership is what matters).
_FLAG_ORDER = [
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
]


def _claimed_parts(obs: VLMObservation) -> list[str]:
    """Ordered, de-duplicated claimed parts (primary first), excluding 'unknown'."""
    ordered: list[str] = []
    for p in [obs.claimed_object_part, *obs.claimed_parts]:
        if p and p != "unknown" and p not in ordered:
            ordered.append(p)
    return ordered


# Issue types that are easily confused / co-occur for the same physical damage.
# Used for LENIENT support matching: an observed issue in the same family as the
# claimed one still counts as support, but a clearly different family does not.
_ISSUE_FAMILIES: list[frozenset[str]] = [
    frozenset({"dent", "scratch"}),
    frozenset({"crack", "glass_shatter"}),
    frozenset({"broken_part", "missing_part"}),
    frozenset({"torn_packaging", "crushed_packaging"}),
    frozenset({"water_damage", "stain"}),
]


def _issue_compatible(claimed_issue: str, observed_issue: str) -> bool:
    """Lenient issue-type match. Unknown/none on either side is treated as
    compatible (don't block support on missing info); otherwise the two must be
    equal or share a damage family. Only a clearly different, known family blocks
    support."""
    if claimed_issue in ("", "unknown", "none") or observed_issue in ("", "unknown", "none"):
        return True
    if claimed_issue == observed_issue:
        return True
    return any(
        claimed_issue in fam and observed_issue in fam for fam in _ISSUE_FAMILIES
    )


def _damage_on_claimed_part(
    img: ImageObservation, claimed: list[str], claimed_issue: str
) -> bool:
    if not (img.matches_claim_object and img.damage_present):
        return False
    if not _issue_compatible(claimed_issue, img.observed_issue_type):
        return False
    if claimed:
        # Damage on a named claimed part, OR damage on the visible claimed part
        # where the model couldn't pin a part label (observed_object_part unknown).
        return img.observed_object_part in claimed or (
            img.claimed_part_visible and img.observed_object_part == "unknown"
        )
    return img.claimed_part_visible


def _order_flags(flags: set[str]) -> list[str]:
    ranked = [f for f in _FLAG_ORDER if f in flags]
    extras = sorted(f for f in flags if f not in _FLAG_ORDER)
    return ranked + extras


def _ids(images: list[ImageObservation]) -> list[str]:
    seen: list[str] = []
    for im in images:
        if im.image_id not in seen:
            seen.append(im.image_id)
    return seen


def _safe_part(part: str, claim_object: str) -> str:
    allowed = valid_parts_for(claim_object)
    return part if part in allowed else "unknown"


def _nei_row(
    claim: PreparedClaim,
    *,
    evidence: bool,
    valid_image: bool,
    object_part: str,
    risk_flags: list[str],
    reason: str,
    justification: str,
) -> OutputRow:
    ci = claim.claim_input
    return OutputRow(
        user_id=ci.user_id,
        image_paths=ci.image_paths,
        user_claim=ci.user_claim,
        claim_object=ci.claim_object,
        evidence_standard_met=evidence,
        evidence_standard_met_reason=reason,
        risk_flags=risk_flags,
        issue_type="unknown",
        object_part=object_part,
        claim_status="not_enough_information",
        claim_status_justification=justification,
        supporting_image_ids=[],
        valid_image=valid_image,
        severity="unknown",
    )


def decide(claim: PreparedClaim, observation: VLMObservation | None) -> OutputRow:
    ci = claim.claim_input
    obj = ci.claim_object
    hist = claim.user_history
    hist_flags = set(hist.flag_list) if hist else set()

    # --- provider failure / no observation -> safe, reviewable NEI ----------
    if observation is None or not observation.images:
        risk = set()
        if "user_history_risk" in hist_flags:
            risk.add("user_history_risk")
        risk.add("manual_review_required")
        if not claim.has_usable_image:
            risk.add("damage_not_visible")
        return _nei_row(
            claim,
            evidence=False,
            valid_image=False,
            object_part="unknown",
            risk_flags=_order_flags(risk),
            reason="No usable image evidence was available to evaluate the claim.",
            justification=(
                "The submitted image set could not be analyzed, so the claim "
                "cannot be verified and is sent for manual review."
            ),
        )

    obs = observation
    images = obs.images
    claimed = _claimed_parts(obs)
    claimed_primary = claimed[0] if claimed else obs.claimed_object_part

    # --- base risk flags from observations ----------------------------------
    risk: set[str] = set()
    for im in images:
        risk.update(f for f in im.quality_flags if f in _QUALITY)
        risk.update(f for f in im.authenticity_flags if f in _AUTHENTICITY)
        if im.embedded_text_present:
            risk.add("text_instruction_present")
    if obs.injection_text_in_claim:
        risk.add("text_instruction_present")

    # --- valid_image: a real, non-manipulated, not-obstructed image exists --
    valid_image = claim.has_usable_image and any(
        not im.authenticity_flags and "cropped_or_obstructed" not in im.quality_flags
        for im in images
    )

    # --- assessability gate (evidence_standard_met) -------------------------
    assessable = any(
        im.claimed_part_visible or im.damage_present or not im.matches_claim_object
        for im in images
    )

    if not assessable:
        # Claimed part not visible and nothing identifiable -> not enough info.
        risk.add("damage_not_visible")
        if "user_history_risk" in hist_flags:
            risk.add("user_history_risk")
        if "manual_review_required" in hist_flags:
            risk.add("manual_review_required")
        return _nei_row(
            claim,
            evidence=False,
            valid_image=valid_image,
            object_part=_safe_part(claimed_primary, obj),
            risk_flags=_order_flags(risk),
            reason=(
                f"The claimed {claimed_primary or obj} part is not visible clearly "
                f"enough in the submitted image set to evaluate the claim."
            ),
            justification=(
                f"The submitted image(s) do not show the claimed "
                f"{claimed_primary or obj} clearly enough, so the claim cannot be "
                f"verified."
            ),
        )

    # --- support vs contradict ---------------------------------------------
    support_images = [
        im for im in images if _damage_on_claimed_part(im, claimed, obs.claimed_issue_type)
    ]

    stated_rank = SEVERITY_RANK.get(obs.stated_severity, -1)
    observed_on_part = max_severity([im.observed_severity for im in support_images])
    exaggeration = (
        bool(support_images)
        and stated_rank >= SEVERITY_RANK["high"]
        and SEVERITY_RANK.get(observed_on_part, -1) in (SEVERITY_RANK["none"], SEVERITY_RANK["low"])
    )

    status: str
    decisive: list[ImageObservation]
    issue_type: str
    object_part: str

    if support_images and not exaggeration:
        status = "supported"
        decisive = _strongest(support_images)
        # Support means the image confirms the customer's claimed damage.
        issue_type = (
            obs.claimed_issue_type
            if obs.claimed_issue_type not in ("unknown", "none")
            else decisive[0].observed_issue_type
        )
        object_part = _safe_part(decisive[0].observed_object_part, obj)
        if object_part == "unknown":
            object_part = _safe_part(claimed_primary, obj)
    else:
        status, decisive, issue_type, object_part, extra = _contradict_or_nei(
            obs, images, claimed, claimed_primary, exaggeration, support_images, obj
        )
        risk.update(extra)
        if status == "not_enough_information":
            risk.add("damage_not_visible")
            _add_history(risk, hist_flags)
            return _nei_row(
                claim,
                evidence=True,
                valid_image=valid_image,
                object_part=_safe_part(claimed_primary, obj),
                risk_flags=_order_flags(risk),
                reason="The image set is usable but does not let the claim be confirmed or denied.",
                justification=(
                    f"The image(s) do not provide clear evidence for or against the "
                    f"claimed {claimed_primary or obj} issue."
                ),
            )

    # --- severity -----------------------------------------------------------
    if issue_type == "none":
        severity = "none"
    else:
        # Observed-only: severity reflects what the image shows, never the
        # customer's stated adjectives (max_severity returns "unknown" if none known).
        severity = max_severity([im.observed_severity for im in decisive])

    # --- history flags (additive; never change status) ----------------------
    _add_history(risk, hist_flags)

    # manual_review_required triggers
    auth_on_decisive = any(im.authenticity_flags for im in decisive)
    if (
        "user_history_risk" in risk
        or "manual_review_required" in hist_flags
        or auth_on_decisive
        or "wrong_object" in risk
        or "possible_manipulation" in risk
    ):
        risk.add("manual_review_required")

    # --- justification + reason --------------------------------------------
    ids = _ids(decisive)
    ids_str = ";".join(ids) if ids else "the submitted image(s)"
    justification, reason = _explain(
        status, obj, issue_type, object_part, claimed_primary, ids_str,
        "user_history_risk" in risk,
    )

    return OutputRow(
        user_id=ci.user_id,
        image_paths=ci.image_paths,
        user_claim=ci.user_claim,
        claim_object=obj,
        evidence_standard_met=True,
        evidence_standard_met_reason=reason,
        risk_flags=_order_flags(risk),
        issue_type=issue_type if issue_type in ISSUE_TYPES else "unknown",
        object_part=object_part,
        claim_status=status,
        claim_status_justification=justification,
        supporting_image_ids=ids,
        valid_image=valid_image,
        severity=severity,
    )


def _strongest(imgs: list[ImageObservation]) -> list[ImageObservation]:
    """Decisive image(s): the most severe first; keep just the top one."""
    ranked = sorted(imgs, key=lambda im: SEVERITY_RANK.get(im.observed_severity, -1), reverse=True)
    return [ranked[0]] if ranked else []


def _contradict_or_nei(
    obs: VLMObservation,
    images: list[ImageObservation],
    claimed: list[str],
    claimed_primary: str,
    exaggeration: bool,
    support_images: list[ImageObservation],
    obj: str,
) -> tuple[str, list[ImageObservation], str, str, set[str]]:
    """Resolve the non-supported branch into contradicted (with reason) or NEI."""
    extra: set[str] = set()

    if exaggeration:
        decisive = _strongest(support_images)
        extra.add("claim_mismatch")
        part = _safe_part(decisive[0].observed_object_part, obj) if decisive else "unknown"
        return "contradicted", decisive, decisive[0].observed_issue_type, part, extra

    wrong_object_imgs = [
        im for im in images if not im.matches_claim_object and im.object_in_image
    ]
    part_visible_no_damage = [
        im for im in images
        if im.matches_claim_object and im.claimed_part_visible and not im.damage_present
    ]
    damage_other_part = [
        im for im in images
        if im.matches_claim_object and im.damage_present and im.observed_object_part not in claimed
    ]

    if wrong_object_imgs:
        extra.update({"wrong_object", "claim_mismatch"})
        return "contradicted", wrong_object_imgs[:1], "unknown", "unknown", extra

    if part_visible_no_damage:
        # Claimed part is clearly visible with no matching damage -> contradicted.
        decisive = part_visible_no_damage[:1]
        extra.add("damage_not_visible")
        part = _safe_part(decisive[0].observed_object_part, obj)
        if part == "unknown":
            part = _safe_part(claimed_primary, obj)
        return "contradicted", decisive, "none", part, extra

    if damage_other_part:
        # Prominent damage, but on a different part than claimed -> contradicted.
        decisive = _strongest(damage_other_part)
        extra.update({"claim_mismatch", "wrong_object_part"})
        part = _safe_part(decisive[0].observed_object_part, obj)
        return "contradicted", decisive, decisive[0].observed_issue_type, part, extra

    return "not_enough_information", [], "unknown", _safe_part(claimed_primary, obj), extra


def _add_history(risk: set[str], hist_flags: set[str]) -> None:
    if "user_history_risk" in hist_flags:
        risk.add("user_history_risk")
    if "manual_review_required" in hist_flags:
        risk.add("manual_review_required")


def _explain(
    status: str,
    obj: str,
    issue_type: str,
    object_part: str,
    claimed_primary: str,
    ids_str: str,
    history_risk: bool,
) -> tuple[str, str]:
    if status == "supported":
        j = f"Image(s) {ids_str} show {issue_type} on the {object_part}, supporting the claim."
        r = f"The {object_part} is visible in the image set, sufficient to assess the claim."
    elif status == "contradicted":
        if issue_type == "none":
            j = (
                f"The {object_part} is visible in image(s) {ids_str} but shows no "
                f"matching damage, so the claim is contradicted."
            )
        elif object_part == "unknown":
            j = (
                f"Image(s) {ids_str} show a different object or no clear match for the "
                f"claimed {obj}, so the claim is contradicted."
            )
        else:
            j = (
                f"Image(s) {ids_str} show {issue_type} on the {object_part}, which does "
                f"not match the claim, so it is contradicted."
            )
        r = "The image set is clear enough to assess the claim."
    else:
        j = f"The claimed {claimed_primary or obj} could not be verified from the images."
        r = f"The claimed {claimed_primary or obj} is not assessable from the image set."

    if history_risk:
        j += " User history adds risk context but did not change the visual decision."
    return j, r
