import httpx
import pytest

from f1_ml_predictor.sources.jolpica import JolpicaClient, JolpicaError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def page(offset: int, total: int, items: list[object], limit: int = 100) -> dict[str, object]:
    return {
        "MRData": {
            "limit": str(limit),
            "offset": str(offset),
            "total": str(total),
            "RaceTable": {"Races": items},
        }
    }


def test_paginates_by_result_limit_with_custom_user_agent() -> None:
    offsets: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ergast/f1/2025/1/results/"
        assert request.url.params["limit"] == "100"
        assert request.headers["User-Agent"] == "test-predictor/2.0"
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(200, json=page(offset, 101, [{"round": "1", "offset": offset}]))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = JolpicaClient(http_client, user_agent="test-predictor/2.0")
        races = client.fetch_collection("2025/1/results/", "RaceTable", "Races")

    assert offsets == [0, 100]
    assert [race["offset"] for race in races] == [0, 100]


def test_retries_429_and_5xx_with_bounded_attempts() -> None:
    clock = FakeClock()
    responses = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(503),
        httpx.Response(200, json=page(0, 1, [{"round": "1"}])),
    ]
    with httpx.Client(transport=httpx.MockTransport(lambda _: responses.pop(0))) as http_client:
        client = JolpicaClient(http_client, clock=clock.clock, sleep=clock.sleep)
        assert client.fetch_collection("2025/races/", "RaceTable", "Races") == [{"round": "1"}]
    assert clock.sleeps == [2.0, 1.0]
    assert responses == []


def test_stops_after_three_retryable_errors() -> None:
    clock = FakeClock()
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = JolpicaClient(http_client, clock=clock.clock, sleep=clock.sleep)
        with pytest.raises(JolpicaError, match="HTTP 500"):
            client.fetch_collection("2025/races/", "RaceTable", "Races")
    assert calls == 3


def test_timeout_is_bounded_and_reported() -> None:
    clock = FakeClock()
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.extensions["timeout"]["read"] == 10.0
        raise httpx.ReadTimeout("slow", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = JolpicaClient(http_client, clock=clock.clock, sleep=clock.sleep)
        with pytest.raises(JolpicaError, match="timed out"):
            client.fetch_collection("2025/races/", "RaceTable", "Races")
    assert calls == 3


def test_transient_connection_error_is_retried() -> None:
    clock = FakeClock()
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(200, json=page(0, 1, [{"round": "1"}]))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = JolpicaClient(http_client, clock=clock.clock, sleep=clock.sleep)
        assert client.fetch_collection("2025/races/", "RaceTable", "Races") == [{"round": "1"}]
    assert calls == 2


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "MRData"),
        ({"MRData": {"total": "oops"}}, "total"),
        (page(0, 2, []), "Empty page"),
        (page(0, 1, ["not an object"]), "Non-object"),
        (page(1, 1, [{"round": "1"}]), "pagination metadata"),
        ({"MRData": {"limit": "100", "offset": "0", "total": "0"}}, "RaceTable"),
    ],
)
def test_rejects_malformed_envelopes(payload: object, match: str) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as http_client:
        with pytest.raises(JolpicaError, match=match):
            JolpicaClient(http_client).fetch_collection("2025/races/", "RaceTable", "Races")


def test_rejects_non_json_and_non_retryable_http_error() -> None:
    for response, message in [
        (httpx.Response(200, content=b"bad json"), "Invalid JSON"),
        (httpx.Response(404), "HTTP 404"),
    ]:
        with httpx.Client(
            transport=httpx.MockTransport(lambda _, response=response: response)
        ) as http_client:
            with pytest.raises(JolpicaError, match=message):
                JolpicaClient(http_client).fetch_collection("2025/races/", "RaceTable", "Races")


def test_local_burst_and_hourly_limits() -> None:
    clock = FakeClock()
    count = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json=page(offset, 501, [{"id": offset}], limit=1))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = JolpicaClient(http_client, clock=clock.clock, sleep=clock.sleep)
        assert len(client.fetch_collection("2025/races/", "RaceTable", "Races")) == 501

    assert count == 501
    assert clock.sleeps[0] == 1.0
    assert clock.now >= 3600.0


@pytest.mark.parametrize("path", ["https://example.com/", "../races/", "races", "races/?x=1"])
def test_rejects_unsafe_or_invalid_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative Jolpica endpoint"):
        JolpicaClient().fetch_collection(path, "RaceTable", "Races")
