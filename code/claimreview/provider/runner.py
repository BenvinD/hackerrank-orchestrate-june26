"""Bounded-concurrency batch runner over claims.

LiteLLM's ``completion`` is blocking I/O, so a thread pool gives real
concurrency for network-bound calls while a fixed worker count caps RPM/TPM
pressure. Results are returned in the original claim order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..schema import PreparedClaim
from .base import ProviderResult, VLMProvider


def analyze_claims(
    provider: VLMProvider,
    claims: list[PreparedClaim],
    max_concurrency: int = 4,
    on_done: Callable[[int, ProviderResult], None] | None = None,
) -> list[ProviderResult]:
    """Analyze every claim, preserving input order.

    ``on_done(index, result)`` is invoked as each claim completes (for progress
    reporting). Exceptions inside a worker are converted to an error result so a
    single failure never aborts the batch.
    """
    results: list[ProviderResult | None] = [None] * len(claims)
    workers = max(1, max_concurrency)

    def _work(idx: int, claim: PreparedClaim) -> tuple[int, ProviderResult]:
        try:
            return idx, provider.analyze(claim)
        except Exception as exc:  # noqa: BLE001 - isolate per-claim failures
            return idx, ProviderResult(
                user_id=claim.claim_input.user_id,
                error=f"runner_error: {type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, i, c) for i, c in enumerate(claims)]
        for fut in futures:
            idx, res = fut.result()
            results[idx] = res
            if on_done is not None:
                on_done(idx, res)

    return [r for r in results if r is not None]
