"""Tiny content-addressed disk cache for provider results.

Stores one JSON file per cache key. Keeps the model-calling layer from
re-billing identical inputs across reruns and across model comparisons.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path


class DiskCache:
    def __init__(self, directory: str | Path, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        final = self._path(key)
        # Unique temp name per writer so concurrent workers handling the same
        # cache key can't clobber each other's temp file. os.replace is atomic.
        tmp = final.with_name(
            f"{key}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
            os.replace(tmp, final)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
