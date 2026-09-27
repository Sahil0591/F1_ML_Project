"""Bounded JSON requests with source-specific local request budgets."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Self

import httpx


class SourceError(RuntimeError):
    """An external response could not be used."""


class JsonSourceClient:
    def __init__(
        self,
        limits: tuple[tuple[int, float], ...],
        client: httpx.Client | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limits = limits
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self._clock = clock
        self._sleep = sleep
        self._requests: deque[float] = deque()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def _throttle(self) -> None:
        while True:
            now = self._clock()
            longest = max(period for _, period in self._limits)
            while self._requests and self._requests[0] <= now - longest:
                self._requests.popleft()
            delay = 0.0
            for capacity, period in self._limits:
                recent = [sent for sent in self._requests if sent > now - period]
                if len(recent) >= capacity:
                    delay = max(delay, recent[-capacity] + period - now)
            if delay <= 0:
                self._requests.append(now)
                return
            self._sleep(delay)

    def get_json(self, url: str, params: dict[str, str | int | float]) -> Any:
        for attempt in range(3):
            self._throttle()
            try:
                response = self._client.get(
                    url,
                    params=params,
                    timeout=10,
                    headers={"User-Agent": "f1-ml-predictor/0.1.0"},
                )
            except httpx.RequestError as exc:
                if attempt == 2:
                    raise SourceError("source request failed after three attempts") from exc
                self._sleep(0.5 * 2**attempt)
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == 2:
                    raise SourceError(f"source returned HTTP {response.status_code}")
                try:
                    delay = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    delay = 0.5 * 2**attempt
                if not math.isfinite(delay) or delay < 0:
                    delay = 0.5 * 2**attempt
                self._sleep(delay)
                continue
            try:
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, ValueError) as exc:
                raise SourceError("source returned an invalid or unsuccessful response") from exc
        raise AssertionError("unreachable retry loop")
