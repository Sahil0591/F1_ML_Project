from pathlib import Path

from f1_ml_predictor.paths import StoragePaths


def test_storage_zones_are_separate_and_lazy(tmp_path: Path) -> None:
    paths = StoragePaths(tmp_path)
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.normalized == tmp_path / "data" / "normalized"
    assert paths.features == tmp_path / "data" / "features"
    assert paths.models == tmp_path / "models"
    assert paths.predictions == tmp_path / "data" / "predictions"
    assert list(tmp_path.iterdir()) == []
