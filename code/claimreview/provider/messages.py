"""Build vision chat messages from a PreparedClaim, and a content-based cache key.

LiteLLM accepts OpenAI-style multimodal content blocks and normalizes them for
Anthropic / Gemini under the hood. The ``provider`` argument is kept as an
explicit hook for the rare per-provider payload tweak.
"""

from __future__ import annotations

import hashlib

from ..config import ModelConfig
from ..prompts import build_system_prompt, build_user_text
from ..schema import PreparedClaim

_SEP = "\x1f"


def build_messages(
    claim: PreparedClaim,
    prompt_version: str = "v1",
    provider: str | None = None,
) -> list[dict]:
    """Construct [system, user(text + images)] messages for a single claim."""
    system = build_system_prompt(prompt_version)
    user_text = build_user_text(claim)

    content: list[dict] = [{"type": "text", "text": user_text}]
    for img in claim.usable_images:
        content.append({"type": "image_url", "image_url": {"url": img.data_url}})

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def cache_key(
    claim: PreparedClaim,
    model_cfg: ModelConfig,
    prompt_version: str = "v1",
) -> str:
    """Stable key over everything that affects the model's answer.

    Uses image *content hashes* (not the base64 bytes) so the key stays small and
    identical inputs reuse the cached result across runs and across the two-model
    comparison (when the model differs, the key differs, as it should).
    """
    parts = [
        model_cfg.provider,
        model_cfg.model,
        prompt_version,
        build_system_prompt(prompt_version),
        build_user_text(claim),
    ]
    parts.extend(sorted(img.content_hash or "" for img in claim.usable_images))
    digest = hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()
    return digest
