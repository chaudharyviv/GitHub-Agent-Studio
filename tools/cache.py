"""
Response cache for the GitHub client.

Entries hold the parsed JSON body plus the ETag so an expired entry can be
revalidated with ``If-None-Match`` (a 304 does not count against the rate limit).
Memory is always used; a disk directory is optional and survives restarts.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheEntry:
    data: Any
    etag: Optional[str]
    fetched_at: float

    def is_fresh(self, ttl: float, now: Optional[float] = None) -> bool:
        return ((now if now is not None else time.time()) - self.fetched_at) < ttl


class ResponseCache:
    """Thread-safe in-memory cache with optional JSON-file persistence."""

    def __init__(self, disk_dir: Optional[str | Path] = None, max_entries: int = 512):
        self._mem: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._disk_dir = Path(disk_dir) if disk_dir else None
        if self._disk_dir:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._mem.get(key)
        if entry is not None or self._disk_dir is None:
            return entry
        entry = self._read_disk(key)
        if entry is not None:
            with self._lock:
                self._mem[key] = entry
        return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            self._mem[key] = entry
            while len(self._mem) > self._max_entries:
                self._mem.pop(next(iter(self._mem)))  # evict oldest inserted
        if self._disk_dir is not None:
            self._write_disk(key, entry)

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
        if self._disk_dir is not None:
            for f in self._disk_dir.glob("*.json"):
                f.unlink(missing_ok=True)

    # -- disk helpers (best effort: a broken cache must never break a tool) --

    def _path(self, key: str) -> Path:
        assert self._disk_dir is not None
        return self._disk_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _read_disk(self, key: str) -> Optional[CacheEntry]:
        try:
            raw = json.loads(self._path(key).read_text(encoding="utf-8"))
            return CacheEntry(raw["data"], raw.get("etag"), raw["fetched_at"])
        except (OSError, ValueError, KeyError):
            return None

    def _write_disk(self, key: str, entry: CacheEntry) -> None:
        try:
            payload = {"data": entry.data, "etag": entry.etag, "fetched_at": entry.fetched_at}
            self._path(key).write_text(json.dumps(payload), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            pass
