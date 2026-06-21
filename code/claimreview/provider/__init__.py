"""Layer 2: the VLM provider.

One structured model call per claim behind a provider-agnostic interface,
backed by LiteLLM so the active model is chosen in config.yaml. The model
returns observations (see claimreview.observation); the rule layer decides.
"""

from .base import ProviderResult, VLMProvider
from .messages import build_messages, cache_key
from .litellm_provider import LiteLLMProvider
from .runner import analyze_claims

__all__ = [
    "ProviderResult",
    "VLMProvider",
    "build_messages",
    "cache_key",
    "LiteLLMProvider",
    "analyze_claims",
]
