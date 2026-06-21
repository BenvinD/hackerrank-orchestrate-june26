"""Typed records that flow through the pipeline.

Input records mirror the dataset CSVs. ``PreparedClaim`` is the fully-joined,
image-encoded unit that the VLM/provider layer will consume - one per claim row.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


def parse_image_paths(raw: str) -> list[str]:
    """Split a semicolon-separated ``image_paths`` cell into clean relative paths."""
    if not raw:
        return []
    return [p.strip() for p in raw.split(";") if p.strip()]


def image_id_from_path(path: str) -> str:
    """Image ID == filename without extension (e.g. ``images/.../img_2.jpg`` -> ``img_2``)."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


class ClaimInput(BaseModel):
    """One row of claims.csv / sample_claims.csv (input columns only)."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str

    @property
    def image_path_list(self) -> list[str]:
        return parse_image_paths(self.image_paths)

    @property
    def image_ids(self) -> list[str]:
        return [image_id_from_path(p) for p in self.image_path_list]


class UserHistory(BaseModel):
    """One row of user_history.csv."""

    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: str = "none"
    history_summary: str = ""

    @field_validator(
        "past_claim_count",
        "accept_claim",
        "manual_review_claim",
        "rejected_claim",
        "last_90_days_claim_count",
        mode="before",
    )
    @classmethod
    def _coerce_int(cls, v: object) -> int:
        if v is None or v == "":
            return 0
        return int(str(v).strip())

    @property
    def flag_list(self) -> list[str]:
        if not self.history_flags or self.history_flags.strip().lower() == "none":
            return []
        return [f.strip() for f in self.history_flags.split(";") if f.strip()]


class EvidenceRequirement(BaseModel):
    """One row of evidence_requirements.csv."""

    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str

    def applies_to_object(self, obj: str) -> bool:
        return self.claim_object == "all" or self.claim_object == obj


class EncodedImage(BaseModel):
    """A single submitted image after resolution, downscale and base64 encoding."""

    image_id: str
    rel_path: str
    abs_path: str
    exists: bool
    media_type: str = "image/jpeg"
    width: int | None = None
    height: int | None = None
    orig_width: int | None = None
    orig_height: int | None = None
    content_hash: str | None = None
    data_url: str | None = Field(default=None, repr=False)
    error: str | None = None

    @property
    def usable(self) -> bool:
        """True when the image was found and successfully encoded for a model call."""
        return self.exists and self.error is None and self.data_url is not None


class PreparedClaim(BaseModel):
    """Fully preprocessed claim: input + joined history + relevant requirements + images.

    This is the contract handed to layer 2 (the VLM provider). It carries
    everything needed to build a single structured model call, with no further
    disk access required.
    """

    claim_input: ClaimInput
    user_history: UserHistory | None
    evidence_requirements: list[EvidenceRequirement]
    images: list[EncodedImage]

    @property
    def usable_images(self) -> list[EncodedImage]:
        return [img for img in self.images if img.usable]

    @property
    def has_usable_image(self) -> bool:
        return any(img.usable for img in self.images)
