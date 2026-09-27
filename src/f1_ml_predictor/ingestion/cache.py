"""Versioned, content-addressed storage for unnormalized source collections."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from f1_ml_predictor.time import require_utc


@dataclass(frozen=True, slots=True)
class CachedCollection:
    items: list[dict[str, Any]]
    sha256: str
    retrieved_at: datetime
    request_path: str


class RawCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, relative_key: str) -> CachedCollection | None:
        directory = self._directory(relative_key)
        manifest_path = directory / "latest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = manifest["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("invalid cache digest")
            return self._read_version(directory / f"{digest}.json", digest)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid raw cache at {manifest_path}") from exc

    def save(
        self,
        relative_key: str,
        items: list[dict[str, Any]],
        retrieved_at: datetime,
        request_path: str,
    ) -> CachedCollection:
        directory = self._directory(relative_key)
        require_utc(retrieved_at, "retrieved_at")
        serialized = _canonical_items(items)
        digest = hashlib.sha256(serialized).hexdigest()
        current = self.load(relative_key)
        if current is not None and current.sha256 == digest:
            return current

        version_path = directory / f"{digest}.json"
        if version_path.exists():
            collection = self._read_version(version_path, digest)
        else:
            collection = CachedCollection(
                json.loads(serialized), digest, retrieved_at, request_path
            )
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_write(version_path, _serialize_version(collection))

        _atomic_write(directory / "latest.json", _canonical_json({"sha256": digest}))
        return collection

    def _directory(self, relative_key: str) -> Path:
        parts = relative_key.replace("\\", "/").split("/")
        if not relative_key or any(part in ("", ".", "..") for part in parts):
            raise ValueError("relative_key must be a nonempty relative path")
        if ":" in parts[0]:
            raise ValueError("relative_key must be a relative path")
        root = self.root.resolve()
        directory = (root / Path(*parts)).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("relative_key escapes cache root")
        return directory

    @staticmethod
    def _read_version(path: Path, digest: str) -> CachedCollection:
        try:
            version = json.loads(path.read_text(encoding="utf-8"))
            items = version["items"]
            retrieved_at = datetime.fromisoformat(version["retrieved_at"])
            request_path = version["request_path"]
            if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                raise ValueError("invalid cache items")
            if not isinstance(request_path, str):
                raise ValueError("invalid request path")
            require_utc(retrieved_at, "retrieved_at")
            if hashlib.sha256(_canonical_items(items)).hexdigest() != digest:
                raise ValueError("raw cache content hash mismatch")
            return CachedCollection(items, digest, retrieved_at, request_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid raw cache version at {path}") from exc


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_items(items: list[dict[str, Any]]) -> bytes:
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("items must be a list of objects")
    return _canonical_json(items)


def _serialize_version(collection: CachedCollection) -> bytes:
    return _canonical_json(
        {
            "items": collection.items,
            "retrieved_at": collection.retrieved_at.isoformat(),
            "request_path": collection.request_path,
        }
    )


def _atomic_write(path: Path, content: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".tmp-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
