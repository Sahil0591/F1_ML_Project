"""Two-stage shadow orders and the live forecast fallback."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np

from f1_ml_predictor.prediction.joint import marginals
from f1_ml_predictor.prediction.shadow import live_weather, two_stage_orders


def test_two_stage_orders_are_coherent_permutations() -> None:
    utility = np.array([2.0, 1.0, 0.0, -1.0])
    dnf = np.array([0.1, 0.0, 0.2, 0.0])
    order, retired = two_stage_orders(
        utility, dnf, 0.5, 1.5, draws=4000, rng=np.random.default_rng(1)
    )
    assert (np.sort(order, axis=1) == np.arange(4)).all()
    values = marginals(order, retired)
    assert np.allclose(values["finish"].sum(axis=1), 1.0)
    assert (values["winner"] <= values["podium"] + 1e-12).all()
    # A retired driver never wins while anyone is still classified.
    first = order[:, 0]
    classified = ~retired.all(axis=1)
    assert not retired[np.arange(len(first)), first][classified].any()


def test_a_colder_first_pick_concentrates_the_winner_only() -> None:
    utility = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    dnf = np.zeros(5)
    flat = marginals(
        *two_stage_orders(utility, dnf, 1.0, 1.0, draws=20000, rng=np.random.default_rng(2))
    )
    sharp = marginals(
        *two_stage_orders(utility, dnf, 0.5, 1.0, draws=20000, rng=np.random.default_rng(2))
    )
    assert sharp["winner"][0] > flat["winner"][0] + 0.1
    # The bottom of the order is drawn with the unchanged rest temperature.
    assert abs(sharp["finish"][4, 4] - flat["finish"][4, 4]) < 0.05


def test_live_weather_reports_an_unavailable_run(tmp_path: Path) -> None:
    cutoff = datetime(2026, 11, 7, 19, 0, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["run"] == "2026-11-07T00:00"
        return httpx.Response(400, json={"error": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = live_weather(
            tmp_path, -23.7, -46.7, cutoff, cutoff + timedelta(hours=22), http_client=client
        )
    assert record["status"] == "unavailable"
    assert record["available_at"] <= cutoff
