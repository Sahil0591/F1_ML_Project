import os
import subprocess
import sys
from pathlib import Path

import pytest

from f1_ml_predictor.trust.locking import LockUnavailableError, advisory_lock


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return environment


def test_existing_stale_file_does_not_own_a_lock(tmp_path: Path) -> None:
    path = tmp_path / "previous.lock"
    path.write_text("12345:former process")
    with advisory_lock(path):
        assert path.exists()
    assert path.read_text() == "12345:former process"


def test_lock_file_is_never_unlinked_or_replaced_on_release(tmp_path: Path) -> None:
    path = tmp_path / "stable.lock"
    with advisory_lock(path):
        original_identity = (path.stat().st_dev, path.stat().st_ino)
        with pytest.raises(LockUnavailableError):
            with advisory_lock(path):
                pytest.fail("overlapping writer acquired the lock")
    assert path.exists()
    assert (path.stat().st_dev, path.stat().st_ino) == original_identity
    with advisory_lock(path):
        assert (path.stat().st_dev, path.stat().st_ino) == original_identity
    assert path.exists()


def test_competing_process_is_denied_without_waiting(tmp_path: Path) -> None:
    path = tmp_path / "competing.lock"
    script = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "from f1_ml_predictor.trust.locking import advisory_lock, LockUnavailableError",
            "try:",
            "    with advisory_lock(Path(sys.argv[1])):",
            "        raise RuntimeError('competing process acquired the lock')",
            "except LockUnavailableError:",
            "    print('denied')",
        ]
    )
    with advisory_lock(path):
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            env=child_environment(),
        )
    assert result.stdout.strip() == "denied"
    assert path.exists()


def test_process_crash_releases_lock_and_preserves_stable_path(tmp_path: Path) -> None:
    path = tmp_path / "crashed.lock"
    script = "\n".join(
        [
            "from pathlib import Path",
            "import os, sys",
            "from f1_ml_predictor.trust.locking import advisory_lock",
            "with advisory_lock(Path(sys.argv[1])):",
            "    os._exit(17)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        env=child_environment(),
    )
    assert result.returncode == 17
    original_identity = (path.stat().st_dev, path.stat().st_ino)
    with advisory_lock(path):
        assert (path.stat().st_dev, path.stat().st_ino) == original_identity
    assert path.exists()


def test_exception_releases_ownership(tmp_path: Path) -> None:
    path = tmp_path / "exception.lock"
    with pytest.raises(RuntimeError, match="tick failed"):
        with advisory_lock(path):
            raise RuntimeError("tick failed")
    with advisory_lock(path):
        assert path.exists()
