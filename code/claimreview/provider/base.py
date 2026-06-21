"""Provider interface and the result envelope it returns."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..observation import VLMObservation
from ..schema import PreparedClaim


class ProviderResult(BaseModel):
    """Outcome of analysing one claim: the observation plus call accounting.

    ``observation`` is None only when the call/parse failed irrecoverably; the
    rule layer then falls back to a safe not_enough_information outcome.
    """

    user_id: str
    observation: VLMObservation | None = None
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    latency_s: float = 0.0
    error: str | None = None
    raw_text: str | None = None


class VLMProvider(ABC):
    """Provider-agnostic single-claim analyzer."""

    @abstractmethod
    def analyze(self, claim: PreparedClaim) -> ProviderResult:
        """Run one structured VLM call for a claim and return observations."""
        raise NotImplementedError
