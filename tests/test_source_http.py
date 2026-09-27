from __future__ import annotations

import httpx
import pytest

from f1_ml_predictor.sources.http import JsonSourceClient, SourceError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


def test_retries_transient_status_with_retry_after_and_stops_after_three() -> None:
    attempts = 0
    clock = FakeClock()

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        if attempts == 2:
            return httpx.Response(503, headers={"Retry-After": "invalid"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as http:
        source = JsonSourceClient(((10, 60.0),), http, clock=clock, sleep=clock.sleep)
        assert source.get_json("https://example.test/data", {"page": 1}) == {"ok": True}
    assert attempts == 3
    assert clock.sleeps == [0.25, 1.0]

    attempts = 0

    def unavailable(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as http:
        source = JsonSourceClient(((10, 60.0),), http, clock=clock, sleep=clock.sleep)
        with pytest.raises(SourceError, match="HTTP 503"):
            source.get_json("https://example.test/data", {})
    assert attempts == 3


def test_request_errors_are_retried_only_three_times() -> None:
    attempts = 0
    clock = FakeClock()

    def timeout(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out")

    with httpx.Client(transport=httpx.MockTransport(timeout)) as http:
        source = JsonSourceClient(((10, 60.0),), http, clock=clock, sleep=clock.sleep)
        with pytest.raises(SourceError, match="three attempts"):
            source.get_json("https://example.test/data", {})
    assert attempts == 3
    assert clock.sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(404, json={"error": "missing"}),
    ],
)
def test_malformed_json_and_unsuccessful_status_raise_source_error(
    response: httpx.Response,
) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _request: response)) as http:
        source = JsonSourceClient(((10, 60.0),), http)
        with pytest.raises(SourceError, match="invalid or unsuccessful"):
            source.get_json("https://example.test/data", {})


def test_rate_budget_waits_until_oldest_request_expires() -> None:
    requests: list[float] = []
    clock = FakeClock()

    def respond(_request: httpx.Request) -> httpx.Response:
        requests.append(clock.now)
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        source = JsonSourceClient(((2, 10.0),), http, clock=clock, sleep=clock.sleep)
        for _ in range(3):
            source.get_json("https://example.test/data", {})
    assert requests == [0.0, 0.0, 10.0]
    assert clock.sleeps == [10.0]
