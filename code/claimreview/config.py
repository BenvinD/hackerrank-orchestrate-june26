"""Configuration loading.

Loads ``config.yaml`` into typed settings and resolves all paths against the
repo root so the solution runs identically regardless of the current working
directory. API keys are never stored here - only the *name* of the env var to
read at call time.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

# repo_root / code / claimreview / config.py  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "code" / "config.yaml"


class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key_env: str | None = None
    # Authoritative pricing (USD per million tokens). When set, the provider uses
    # these for cost accounting instead of LiteLLM's built-in table.
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Cost in USD from config pricing, or None if pricing is not configured."""
        if self.input_per_mtok is None or self.output_per_mtok is None:
            return None
        return (
            prompt_tokens / 1_000_000 * self.input_per_mtok
            + completion_tokens / 1_000_000 * self.output_per_mtok
        )


class PathsConfig(BaseModel):
    dataset_dir: str = "dataset"
    claims_csv: str = "dataset/claims.csv"
    sample_csv: str = "dataset/sample_claims.csv"
    user_history_csv: str = "dataset/user_history.csv"
    evidence_requirements_csv: str = "dataset/evidence_requirements.csv"
    output_csv: str = "output.csv"


class ImageConfig(BaseModel):
    max_long_side: int = 1024
    jpeg_quality: int = 85
    format: str = "JPEG"


class RuntimeConfig(BaseModel):
    temperature: float = 0
    max_concurrency: int = 4
    max_retries: int = 3
    request_timeout_s: int = 90


class CacheConfig(BaseModel):
    enabled: bool = True
    dir: str = ".cache"


class Config(BaseModel):
    active_model: str = "claude-sonnet"
    models: dict[str, ModelConfig] = {}
    paths: PathsConfig = PathsConfig()
    image: ImageConfig = ImageConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    cache: CacheConfig = CacheConfig()
    prompt_version: str = "v1"

    def resolve(self, rel_or_abs: str) -> Path:
        """Resolve a config path against the repo root (absolute paths pass through)."""
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def active_model_config(self) -> ModelConfig:
        if self.active_model not in self.models:
            raise KeyError(
                f"active_model '{self.active_model}' not found in models: "
                f"{sorted(self.models)}"
            )
        return self.models[self.active_model]


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config.yaml`` (defaults to ``code/config.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
