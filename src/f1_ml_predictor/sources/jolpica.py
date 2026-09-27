"""Bounded, rate-limited access to Jolpica's Ergast-compatible API."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

import httpx

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100
MAX_ATTEMPTS = 3


class JolpicaError(RuntimeError):
    """A request or response could not be used safely."""


class JolpicaClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        user_agent: str = "f1-ml-predictor/0.1.0",
        timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("user_agent must identify the application")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None
        self._user_agent = user_agent
        self._timeout = timeout
        self._clock = clock
        self._sleep = sleep
        self._requests: deque[float] = deque()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JolpicaClient:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def fetch_collection(
        self, path: str, table_key: str, collection_key: str
    ) -> list[dict[str, Any]]:
        """Fetch every page of a named MRData table collection."""
        if (
            not path
            or not path.endswith(("/", ".json"))
            or "://" in path
            or "?" in path
            or "#" in path
            or ".." in path.split("/")
        ):
            raise ValueError("path must be a relative Jolpica endpoint ending in / or .json")
        if not table_key or not collection_key:
            raise ValueError("table_key and collection_key must be nonempty")

        url = f"{BASE_URL}/{path.lstrip('/')}"
        offset = 0
        rows: list[dict[str, Any]] = []
        while True:
            response = self._get(url, offset)
            try:
                payload = response.json()
            except ValueError as exc:
                raise JolpicaError(f"Invalid JSON from Jolpica at offset {offset}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("MRData"), dict):
                raise JolpicaError(f"Missing MRData object at offset {offset}")
            mrdata = payload["MRData"]
            total = self._count(mrdata.get("total"), "total", offset, allow_zero=True)
            page_limit = self._count(mrdata.get("limit"), "limit", offset)
            page_offset = self._count(mrdata.get("offset"), "offset", offset, allow_zero=True)
            if page_offset != offset or page_limit > PAGE_LIMIT:
                raise JolpicaError(f"Invalid pagination metadata at offset {offset}")
            table = mrdata.get(table_key)
            if not isinstance(table, dict) or not isinstance(table.get(collection_key), list):
                raise JolpicaError(f"Missing {table_key}.{collection_key} list at offset {offset}")
            page = table[collection_key]
            if not all(isinstance(item, dict) for item in page):
                raise JolpicaError(
                    f"Non-object item in {table_key}.{collection_key} at offset {offset}"
                )
            if offset < total and not page:
                raise JolpicaError(f"Empty page before total {total} at offset {offset}")
            rows.extend(page)
            offset += page_limit
            if offset >= total:
                return rows

    @staticmethod
    def _count(value: object, name: str, offset: int, *, allow_zero: bool = False) -> int:
        if not isinstance(value, (int, str)) or isinstance(value, bool):
            raise JolpicaError(f"Invalid MRData.{name} at offset {offset}")
        if isinstance(value, str) and not value.isdecimal():
            raise JolpicaError(f"Invalid MRData.{name} at offset {offset}")
        number = int(value)
        if number < (0 if allow_zero else 1):
            raise JolpicaError(f"Invalid MRData.{name} at offset {offset}")
        return number

    def _throttle(self) -> None:
        while True:
            now = self._clock()
            while self._requests and self._requests[0] <= now - 3600:
                self._requests.popleft()
            recent = [sent for sent in self._requests if sent > now - 1]
            wait_for = 0.0
            if len(recent) >= 4:
                wait_for = max(wait_for, recent[0] + 1 - now)
            if len(self._requests) >= 500:
                wait_for = max(wait_for, self._requests[0] + 3600 - now)
            if wait_for <= 0:
                self._requests.append(now)
                return
            self._sleep(wait_for)

    def _get(self, url: str, offset: int) -> httpx.Response:
        for attempt in range(MAX_ATTEMPTS):
            self._throttle()
            try:
                response = self._client.get(
                    url,
                    params={"limit": PAGE_LIMIT, "offset": offset},
                    headers={"User-Agent": self._user_agent},
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise JolpicaError(f"Jolpica timed out at offset {offset}") from exc
                self._sleep(0.5 * 2**attempt)
                continue
            except httpx.RequestError as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise JolpicaError(f"Jolpica request failed at offset {offset}: {exc}") from exc
                self._sleep(0.5 * 2**attempt)
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == MAX_ATTEMPTS - 1:
                    raise JolpicaError(
                        f"Jolpica returned HTTP {response.status_code} at offset {offset}"
                    )
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = max(0.0, float(retry_after))
                except ValueError:
                    delay = 0.5 * 2**attempt
                self._sleep(delay)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise JolpicaError(
                    f"Jolpica returned HTTP {response.status_code} at offset {offset}"
                ) from exc
            return response
        raise AssertionError("unreachable retry loop")
