import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from f1_ml_predictor.ingestion.cache import RawCache

FIRST = datetime(2025, 5, 1, 10, tzinfo=UTC)
LATER = FIRST + timedelta(hours=1)


def test_cache_hit_preserves_first_retrieval_and_canonical_hash(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    items = [{"b": 2, "a": 1}]
    saved = cache.save("jolpica/2025/1/results", items, FIRST, "/2025/1/results/")
    expected = hashlib.sha256(b'[{"a":1,"b":2}]').hexdigest()
    assert saved.sha256 == expected
    assert cache.load("jolpica/2025/1/results") == saved

    reused = cache.save("jolpica/2025/1/results", [{"a": 1, "b": 2}], LATER, "/new/")
    assert reused == saved
    assert len(list((tmp_path / "jolpica/2025/1/results").glob("*.json"))) == 2


def test_changed_response_creates_version_and_can_reuse_older_version(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    key = "jolpica/2025/1/qualifying"
    first = cache.save(key, [{"position": "1"}], FIRST, "/first/")
    second = cache.save(key, [{"position": "2"}], LATER, "/second/")
    assert first.sha256 != second.sha256
    assert cache.load(key) == second

    restored = cache.save(key, [{"position": "1"}], LATER, "/again/")
    assert restored == first
    assert cache.load(key) == first
    assert len(list((tmp_path / key).glob("*.json"))) == 3


def test_corrupt_content_is_rejected(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    key = "jolpica/2025/1/results"
    saved = cache.save(key, [{"position": "1"}], FIRST, "/results/")
    version_path = tmp_path / key / f"{saved.sha256}.json"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    version["items"] = [{"position": "2"}]
    version_path.write_text(json.dumps(version), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.load(key)


@pytest.mark.parametrize(
    "key",
    ["../outside", "nested/../../outside", "/absolute", "C:/outside", "a\\..\\outside", ""],
)
def test_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    cache = RawCache(tmp_path)
    with pytest.raises(ValueError):
        cache.load(key)
    with pytest.raises(ValueError):
        cache.save(key, [], FIRST, "/")


def test_unchanged_response_does_not_write(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    key = "jolpica/2025/1/results"
    first = cache.save(key, [{"position": "1"}], FIRST, "/results/")
    with patch(
        "f1_ml_predictor.ingestion.cache._atomic_write", side_effect=AssertionError("wrote")
    ):
        assert cache.save(key, [{"position": "1"}], LATER, "/results/") == first


def test_rejects_non_utc_retrieval_time(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        cache.save("jolpica/2025/1/results", [], FIRST.replace(tzinfo=None), "/results/")
