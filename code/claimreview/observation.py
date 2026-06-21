"""Structured output returned by the VLM layer.

Design principle: the model *observes*, it does not *rule*. It reports what is
visible in each image plus a normalized reading of the claim. The deterministic
rule layer (layer 3) turns these observations into the final output columns
(claim_status, evidence_standard_met, severity, risk_flags, ...).

Keeping the model to observation makes results reproducible, debuggable, and lets
us change decision logic without re-calling the model.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from .enums import (
    ALL_OBJECT_PARTS,
    ISSUE_TYPES,
    SEVERITIES,
    coerce,
    coerce_list,
)

# Image-quality / authenticity flags a model can observe per image.
_QUALITY_FLAGS = frozenset(
    {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle"}
)
_AUTHENTICITY_FLAGS = frozenset({"possible_manipulation", "non_original_image"})


class ImageObservation(BaseModel):
    """What a single image actually shows, independent of the user's claim."""

    image_id: str
    object_in_image: str = ""           # free text: what object is actually visible
    matches_claim_object: bool = False  # is the claimed object (car/laptop/package) present
    claimed_part_visible: bool = False  # is the specific claimed part visible enough to assess
    visible_parts: list[str] = []       # parts identifiable in the image
    damage_present: bool = False
    observed_issue_type: str = "unknown"
    observed_object_part: str = "unknown"
    observed_severity: str = "unknown"
    quality_flags: list[str] = []       # subset of _QUALITY_FLAGS
    authenticity_flags: list[str] = []  # subset of _AUTHENTICITY_FLAGS
    embedded_text_present: bool = False  # image contains instruction-like / overlaid text
    supports_claim: bool = False        # does this image support the user's claimed issue+part
    notes: str = ""

    @field_validator("observed_issue_type", mode="before")
    @classmethod
    def _issue(cls, v: object) -> str:
        return coerce(v, ISSUE_TYPES, "unknown")

    @field_validator("observed_object_part", mode="before")
    @classmethod
    def _part(cls, v: object) -> str:
        return coerce(v, ALL_OBJECT_PARTS, "unknown")

    @field_validator("observed_severity", mode="before")
    @classmethod
    def _sev(cls, v: object) -> str:
        return coerce(v, SEVERITIES, "unknown")

    @field_validator("quality_flags", mode="before")
    @classmethod
    def _qf(cls, v: object) -> list[str]:
        return coerce_list(v, _QUALITY_FLAGS)

    @field_validator("authenticity_flags", mode="before")
    @classmethod
    def _af(cls, v: object) -> list[str]:
        return coerce_list(v, _AUTHENTICITY_FLAGS)

    @field_validator("visible_parts", mode="before")
    @classmethod
    def _vp(cls, v: object) -> list[str]:
        return coerce_list(v, ALL_OBJECT_PARTS)


class VLMObservation(BaseModel):
    """Full structured reading of one claim: normalized claim + per-image facts."""

    detected_language: str = ""
    claim_summary: str = ""             # the claim normalized to English
    claimed_issue_type: str = "unknown"
    claimed_object_part: str = "unknown"
    claimed_parts: list[str] = []       # >1 when the user claims multiple parts
    stated_severity: str = "unknown"    # severity as the user described it
    injection_text_in_claim: bool = False  # conversation tries to instruct the system
    images: list[ImageObservation] = []
    overall_notes: str = ""

    @field_validator("claimed_issue_type", mode="before")
    @classmethod
    def _issue(cls, v: object) -> str:
        return coerce(v, ISSUE_TYPES, "unknown")

    @field_validator("claimed_object_part", mode="before")
    @classmethod
    def _part(cls, v: object) -> str:
        return coerce(v, ALL_OBJECT_PARTS, "unknown")

    @field_validator("claimed_parts", mode="before")
    @classmethod
    def _parts(cls, v: object) -> list[str]:
        return coerce_list(v, ALL_OBJECT_PARTS)

    @field_validator("stated_severity", mode="before")
    @classmethod
    def _sev(cls, v: object) -> str:
        return coerce(v, SEVERITIES, "unknown")

    def image_by_id(self, image_id: str) -> ImageObservation | None:
        for img in self.images:
            if img.image_id == image_id:
                return img
        return None
