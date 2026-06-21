"""Assemble fully-preprocessed claims ready for the VLM layer.

Joins each claim to its user history and to the evidence requirements relevant
to its object, then encodes every submitted image. No model calls happen here.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from ..schema import (
    ClaimInput,
    EvidenceRequirement,
    PreparedClaim,
    UserHistory,
)
from .images import encode_image
from .loaders import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
)


def _requirements_for(
    claim_object: str, requirements: list[EvidenceRequirement]
) -> list[EvidenceRequirement]:
    """Requirements that apply to this object (object-specific + ``all``)."""
    return [r for r in requirements if r.applies_to_object(claim_object)]


def prepare_claims(
    claims: list[ClaimInput],
    history_map: dict[str, UserHistory],
    requirements: list[EvidenceRequirement],
    config: Config,
) -> list[PreparedClaim]:
    """Join + encode each claim into a ``PreparedClaim``."""
    dataset_dir = config.resolve(config.paths.dataset_dir)
    prepared: list[PreparedClaim] = []
    for claim in claims:
        images = [
            encode_image(rel, dataset_dir, config.image)
            for rel in claim.image_path_list
        ]
        prepared.append(
            PreparedClaim(
                claim_input=claim,
                user_history=history_map.get(claim.user_id),
                evidence_requirements=_requirements_for(claim.claim_object, requirements),
                images=images,
            )
        )
    return prepared


def prepare_dataset(
    which: str = "test",
    config: Config | None = None,
) -> list[PreparedClaim]:
    """Convenience: load + prepare the ``test`` (claims.csv) or ``sample`` set.

    ``test``   -> dataset/claims.csv         (input-only; produce output.csv)
    ``sample`` -> dataset/sample_claims.csv  (labeled; for evaluation)
    """
    cfg = config or load_config()
    if which == "test":
        claims_path: Path = cfg.resolve(cfg.paths.claims_csv)
    elif which == "sample":
        claims_path = cfg.resolve(cfg.paths.sample_csv)
    else:
        raise ValueError(f"which must be 'test' or 'sample', got {which!r}")

    claims = load_claims(claims_path)
    history_map = load_user_history(cfg.resolve(cfg.paths.user_history_csv))
    requirements = load_evidence_requirements(
        cfg.resolve(cfg.paths.evidence_requirements_csv)
    )
    return prepare_claims(claims, history_map, requirements, cfg)
