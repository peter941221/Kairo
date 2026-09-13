"""Content-addressed runtime cache primitives for Kairo kernels.

The cache deliberately stores opaque bytes: a caller may use it for PTX,
Cubin, a serialized TMA descriptor, or a compiled graph package.  The key
contains every environment dimension that can change generated code, while
the metadata makes misses and invalidation reasons inspectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class KernelCacheKey:
    """Stable identity for one generated-kernel artifact."""

    blueprint_hash: str
    shape: tuple[int, ...]
    driver_version: str
    gpu_capability: str
    template_version: str = "v1"

    def as_dict(self) -> dict[str, object]:
        return {
            "blueprint_hash": self.blueprint_hash,
            "shape": list(self.shape),
            "shape_hash": self.shape_hash,
            "driver_version": self.driver_version,
            "gpu_capability": self.gpu_capability,
            "template_version": self.template_version,
        }

    @property
    def shape_hash(self) -> str:
        encoded = json.dumps(list(self.shape), separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class RuntimeKernelCache:
    """Atomic, inspectable cache for opaque compiled artifacts.

    The API is intentionally small so it can sit below either an AOT package
    loader or a bounded JIT compiler.  `load` never raises for a stale or
    corrupt entry; it records a reason and returns ``None`` so the caller can
    safely fall back to compilation or a reference implementation.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # A per-digest lock prevents a thundering herd from compiling the same
        # artifact when several runtime requests miss at once.
        self._build_locks: dict[str, threading.Lock] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "invalidations": 0,
            "miss_reasons": {},
        }

    def _artifact_path(self, key: KernelCacheKey) -> Path:
        return self.root / f"{key.digest}.bin"

    def _metadata_path(self, key: KernelCacheKey) -> Path:
        return self.root / f"{key.digest}.json"

    def _record_miss(self, reason: str) -> None:
        self._stats["misses"] += 1
        reasons = self._stats["miss_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def _find_fingerprint_mismatch(self, key: KernelCacheKey) -> str | None:
        for metadata_path in self.root.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            old = metadata.get("key") or {}
            if old.get("blueprint_hash") != key.blueprint_hash:
                continue
            if old.get("shape_hash") != key.shape_hash:
                continue
            if old.get("driver_version") != key.driver_version:
                return "driver_version_changed"
            if old.get("gpu_capability") != key.gpu_capability:
                return "gpu_capability_changed"
            if old.get("template_version") != key.template_version:
                return "template_version_changed"
        return None

    def load(self, key: KernelCacheKey) -> bytes | None:
        with self._lock:
            metadata_path = self._metadata_path(key)
            artifact_path = self._artifact_path(key)
            if not metadata_path.exists() or not artifact_path.exists():
                reason = self._find_fingerprint_mismatch(key) or "not_found"
                self._record_miss(reason)
                return None
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                payload = artifact_path.read_bytes()
            except (OSError, json.JSONDecodeError):
                self._record_miss("corrupt_metadata")
                self._stats["invalidations"] += 1
                return None
            expected_hash = metadata.get("payload_sha256")
            if metadata.get("key") != key.as_dict() or hashlib.sha256(payload).hexdigest() != expected_hash:
                self._record_miss("integrity_failure")
                self._stats["invalidations"] += 1
                return None
            self._stats["hits"] += 1
            return payload

    def store(self, key: KernelCacheKey, payload: bytes) -> Path:
        if not isinstance(payload, bytes):
            raise TypeError("cache payload must be bytes")
        metadata = {
            "schema": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "key": key.as_dict(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_size": len(payload),
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            artifact_path = self._artifact_path(key)
            metadata_path = self._metadata_path(key)
            for destination, contents in (
                (artifact_path, payload),
                (metadata_path, (json.dumps(metadata, indent=2) + "\n").encode()),
            ):
                fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=self.root)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(contents)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            self._stats["stores"] += 1
            return artifact_path

    def get_or_build(self, key: KernelCacheKey, builder: Callable[[], bytes]) -> tuple[bytes, bool]:
        """Return ``(payload, cache_hit)`` and single-flight publish misses."""

        payload = self.load(key)
        if payload is not None:
            return payload, True
        with self._lock:
            build_lock = self._build_locks.setdefault(key.digest, threading.Lock())
        with build_lock:
            # Another caller may have completed the build while this caller
            # waited for the per-key lock.
            payload = self.load(key)
            if payload is not None:
                return payload, True
            payload = builder()
            self.store(key, payload)
            return payload, False

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                **self._stats,
                "miss_reasons": dict(self._stats["miss_reasons"]),
            }

    def inspect(self) -> dict[str, object]:
        """Audit persistent entries without changing hit/miss counters."""

        with self._lock:
            entries: list[dict[str, object]] = []
            referenced_artifacts: set[str] = set()
            for metadata_path in sorted(self.root.glob("*.json")):
                digest = metadata_path.stem
                artifact_path = self.root / f"{digest}.bin"
                referenced_artifacts.add(artifact_path.name)
                entry: dict[str, object] = {
                    "digest": digest,
                    "metadata": metadata_path.name,
                    "artifact": artifact_path.name,
                    "valid": False,
                }
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    payload = artifact_path.read_bytes()
                    expected_hash = metadata.get("payload_sha256")
                    actual_hash = hashlib.sha256(payload).hexdigest()
                    entry["key"] = metadata.get("key")
                    entry["payload_size"] = len(payload)
                    entry["valid"] = (
                        metadata.get("schema") == 1
                        and artifact_path.exists()
                        and metadata.get("payload_size") == len(payload)
                        and expected_hash == actual_hash
                    )
                    if not entry["valid"]:
                        entry["reason"] = "integrity_failure"
                except FileNotFoundError:
                    entry["reason"] = "missing_artifact"
                except (OSError, json.JSONDecodeError):
                    entry["reason"] = "corrupt_metadata"
                entries.append(entry)
            for artifact_path in sorted(self.root.glob("*.bin")):
                if artifact_path.name not in referenced_artifacts:
                    entries.append(
                        {
                            "digest": artifact_path.stem,
                            "artifact": artifact_path.name,
                            "valid": False,
                            "reason": "orphan_artifact",
                        }
                    )
            valid = sum(bool(entry["valid"]) for entry in entries)
            return {
                "root": str(self.root),
                "entries": entries,
                "entry_count": len(entries),
                "valid_entries": valid,
                "invalid_entries": len(entries) - valid,
                "total_payload_bytes": sum(int(entry.get("payload_size", 0)) for entry in entries),
            }
