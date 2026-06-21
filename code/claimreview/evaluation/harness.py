"""Run a model over a dataset, decide, and aggregate metrics + operational stats."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from ..config import Config
from ..preprocessing import prepare_dataset
from ..provider import analyze_claims
from ..provider.base import ProviderResult
from ..provider.litellm_provider import LiteLLMProvider
from ..rules import decide
from ..rules.output import OutputRow
from .metrics import Metrics, score


class OpsStats(BaseModel):
    claims: int
    new_calls: int
    cached: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float                 # cost of NEW calls only
    images: int
    avg_latency_s: float
    max_latency_s: float


class EvalRun(BaseModel):
    model_name: str
    provider_model: str
    prompt_version: str
    predictions: list[OutputRow]
    ops: OpsStats
    metrics: Metrics | None = None

    @property
    def label(self) -> str:
        return f"{self.model_name}@{self.prompt_version}"


def _read_expected(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _aggregate(results: list[ProviderResult], images_total: int) -> OpsStats:
    lat = [r.latency_s for r in results] or [0.0]
    return OpsStats(
        claims=len(results),
        new_calls=sum(1 for r in results if not r.cached and not r.error),
        cached=sum(1 for r in results if r.cached),
        errors=sum(1 for r in results if r.error),
        prompt_tokens=sum(r.prompt_tokens for r in results),
        completion_tokens=sum(r.completion_tokens for r in results),
        total_tokens=sum(r.total_tokens for r in results),
        cost_usd=sum(r.cost_usd for r in results if not r.cached),
        images=images_total,
        avg_latency_s=sum(lat) / len(lat),
        max_latency_s=max(lat),
    )


def run_model(
    cfg: Config,
    model_name: str,
    which: str = "sample",
    progress: Callable[[int, ProviderResult], None] | None = None,
    prompt_version: str | None = None,
) -> EvalRun:
    """Analyze + decide every claim for one model; score against labels if available.

    ``prompt_version`` overrides config (so v1 vs v2 can be compared); it also
    changes the cache key, so each version is observed independently.
    """
    if prompt_version is not None and prompt_version != cfg.prompt_version:
        cfg = cfg.model_copy(update={"prompt_version": prompt_version})
    prepared = prepare_dataset(which, cfg)
    images_total = sum(len(p.usable_images) for p in prepared)

    provider = LiteLLMProvider(cfg, model_name)
    results = analyze_claims(provider, prepared, cfg.runtime.max_concurrency, progress)
    predictions = [decide(p, r.observation) for p, r in zip(prepared, results)]

    metrics: Metrics | None = None
    if which == "sample":
        expected = _read_expected(cfg.resolve(cfg.paths.sample_csv))
        metrics = score(predictions, expected)

    return EvalRun(
        model_name=model_name,
        provider_model=f"{provider.model_cfg.provider}/{provider.model_cfg.model}",
        prompt_version=cfg.prompt_version,
        predictions=predictions,
        ops=_aggregate(results, images_total),
        metrics=metrics,
    )
