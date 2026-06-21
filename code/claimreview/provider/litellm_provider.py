"""LiteLLM-backed VLMProvider.

One structured vision call per claim. The model string is built from config
(``provider/model``) so swapping Anthropic / OpenAI / Gemini is a config change.
API keys are read from the environment variable named in config, never stored.
"""

from __future__ import annotations

import json
import os
import re
import time

from ..config import Config
from ..observation import VLMObservation
from ..schema import PreparedClaim
from .base import ProviderResult, VLMProvider
from .cache import DiskCache
from .messages import build_messages, cache_key

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?|\n?```$")


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from model text."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(t[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LiteLLMProvider(VLMProvider):
    def __init__(self, config: Config, model_name: str | None = None) -> None:
        self.config = config
        self.model_name = model_name or config.active_model
        if self.model_name not in config.models:
            raise KeyError(
                f"model '{self.model_name}' not in config.models: "
                f"{sorted(config.models)}"
            )
        self.model_cfg = config.models[self.model_name]
        self.litellm_model = f"{self.model_cfg.provider}/{self.model_cfg.model}"
        self.cache = DiskCache(
            config.resolve(config.cache.dir), enabled=config.cache.enabled
        )

    def _api_key(self) -> str | None:
        env = self.model_cfg.api_key_env
        return os.environ.get(env) if env else None

    def _call(self, messages: list[dict]):
        import litellm
        from litellm import completion

        litellm.suppress_debug_info = True  # quiet the help banners on errors
        rt = self.config.runtime
        return completion(
            model=self.litellm_model,
            messages=messages,
            temperature=rt.temperature,
            timeout=rt.request_timeout_s,
            api_key=self._api_key(),
        )

    def _usage_and_cost(self, resp) -> tuple[int, int, int, float]:
        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))

        # Prefer authoritative config pricing; fall back to LiteLLM's table.
        cfg_cost = self.model_cfg.cost_for(pt, ct)
        if cfg_cost is not None:
            return pt, ct, tt, cfg_cost
        try:
            from litellm import completion_cost

            return pt, ct, tt, float(completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001 - pricing lookup is best-effort
            return pt, ct, tt, 0.0

    def analyze(self, claim: PreparedClaim) -> ProviderResult:
        uid = claim.claim_input.user_id
        key = cache_key(claim, self.model_cfg, self.config.prompt_version)

        cached = self.cache.get(key)
        if cached is not None:
            try:
                res = ProviderResult.model_validate(cached)
                res.cached = True
                res.cost_usd = 0.0  # cache hit = no new spend
                return res
            except Exception:  # noqa: BLE001 - ignore a corrupt cache entry
                pass

        messages = build_messages(
            claim, self.config.prompt_version, self.model_cfg.provider
        )
        rt = self.config.runtime
        start = time.time()
        last_err: str | None = None
        # Accumulate usage/cost across every billable call (retries + JSON repair)
        # so reported tokens/cost reflect real spend, not just the final response.
        acc_pt = acc_ct = acc_tt = 0
        acc_cost = 0.0

        for attempt in range(rt.max_retries + 1):
            try:
                resp = self._call(messages)
                pt, ct, tt, cost = self._usage_and_cost(resp)
                acc_pt += pt; acc_ct += ct; acc_tt += tt; acc_cost += cost
                text = resp.choices[0].message.content or ""
                data = extract_json(text)

                if data is None:  # one in-call repair before counting it a failure
                    repair = messages + [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was not valid JSON. Reply again "
                                "with ONLY the JSON object, no prose or code fences."
                            ),
                        },
                    ]
                    resp = self._call(repair)
                    pt, ct, tt, cost = self._usage_and_cost(resp)
                    acc_pt += pt; acc_ct += ct; acc_tt += tt; acc_cost += cost
                    text = resp.choices[0].message.content or ""
                    data = extract_json(text)
                    if data is None:
                        last_err = "json_parse_failed"
                        continue

                obs = VLMObservation.model_validate(data)
                result = ProviderResult(
                    user_id=uid,
                    observation=obs,
                    model=self.model_cfg.model,
                    provider=self.model_cfg.provider,
                    prompt_tokens=acc_pt,
                    completion_tokens=acc_ct,
                    total_tokens=acc_tt,
                    cost_usd=acc_cost,
                    cached=False,
                    latency_s=time.time() - start,
                    raw_text=text,
                )
                self.cache.set(key, result.model_dump())
                return result

            except Exception as exc:  # noqa: BLE001 - retry transient API/network errors
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < rt.max_retries:
                    time.sleep(min(2 ** attempt, 8))

        return ProviderResult(
            user_id=uid,
            observation=None,
            model=self.model_cfg.model,
            provider=self.model_cfg.provider,
            cached=False,
            latency_s=time.time() - start,
            error=last_err or "unknown_error",
        )
