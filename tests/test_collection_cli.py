import sys

import pytest

from f1_ml_predictor import __main__


@pytest.mark.parametrize("status,failed", [("error", True), ("waiting_for_qualifying", False)])
def test_collection_cli_returns_failure_for_recorded_errors(monkeypatch, tmp_path, status, failed):
    monkeypatch.setattr(sys, "argv", ["f1", "collect-next-race", "--root", str(tmp_path)])
    monkeypatch.setattr(__main__, "scheduler_tick", lambda *args, **kwargs: {"status": status})
    if failed:
        with pytest.raises(SystemExit) as error:
            __main__.main()
        assert error.value.code == 1
    else:
        __main__.main()
