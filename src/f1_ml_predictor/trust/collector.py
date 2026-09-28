"""Fresh bounded API acquisition for immutable prospective evaluation inputs."""

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pyarrow as pa

from f1_ml_predictor.identifiers import EventId
from f1_ml_predictor.sources.http import JsonSourceClient
from f1_ml_predictor.time import require_utc
from f1_ml_predictor.trust.cutoffs import CutoffKind, PredictionCutoff
from f1_ml_predictor.trust.prospective import freeze_bundle

_LIMITS = {
    "api.jolpi.ca": ((4, 1.0), (500, 3600.0)),
    "api.openf1.org": ((3, 1.0), (30, 60.0)),
    "api.open-meteo.com": ((600, 60.0), (5000, 3600.0), (10000, 86400.0)),
}
_ROLES = {
    "event",
    "roster",
    "qualifying",
    "grid",
    "session_metadata",
    "session_laps",
    "session_stints",
    "session_pits",
    "forecast",
    "circuit",
}


def _validate_requests(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("capture plan needs at least one request")
    names: set[str] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("role"), str)
            or item["role"] not in _ROLES
        ):
            raise ValueError("capture requests must declare predictive roles")
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("capture input name must be a safe nonempty identifier")
        if name in {"window", "source_requests"} or name in names:
            raise ValueError("reserved or duplicate capture input name")
        names.add(name)
        if not isinstance(item.get("url"), str):
            raise ValueError("capture request URL must be a string")
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("capture request params must be a JSON object")
        for key, scalar in params.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("capture parameter names must be nonempty strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if any(
                secret in normalized
                for secret in ("auth", "token", "password", "secret", "apikey", "credential")
            ):
                raise ValueError("authenticated capture parameters are not supported")
            if scalar is not None and type(scalar) not in {str, int, float, bool}:
                raise ValueError("capture parameters must be JSON scalars")
            if isinstance(scalar, float) and not math.isfinite(scalar):
                raise ValueError("capture parameters must be finite JSON scalars")
    return value


def collect_weekend(
    plan_path: Path, root: Path, *, http_client: httpx.Client | None = None
) -> Path:
    """Capture now, never replay a missed historical cutoff as a live snapshot.

    The plan names official JSON endpoints and the known race/qualifying decision
    times. Each payload is frozen as canonical JSON in a raw Arrow table. It is
    not automatically a complete roster, published grid, or certified feature set.
    """
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict) or type(plan.get("version")) is not int or plan["version"] != 1:
        raise ValueError("capture plan must be a version 1 object")
    try:
        event = EventId(plan["season"], plan["round"])
        race_start = datetime.fromisoformat(plan["race_start"])
        if race_start.year != event.season:
            raise ValueError("capture event season and race time disagree")
        decision = datetime.fromisoformat(plan["qualifying_decision_at"])
        kind = CutoffKind(plan["cutoff_kind"])
        minutes = plan.get("pre_race_minutes", 60)
        PredictionCutoff(kind, datetime.now(UTC), minutes).validate(race_start, decision)
        requests = _validate_requests(plan["requests"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed capture plan") from exc
    clients: dict[str, JsonSourceClient] = {}
    tables, source_requests = {}, {}
    validated_sessions: set[int] = set()
    try:
        for item in sorted(requests, key=lambda value: value.get("role") != "session_metadata"):
            name, role, url = item["name"], item["role"], item["url"]
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            if (
                parsed.scheme != "https"
                or host not in _LIMITS
                or parsed.username
                or parsed.password
                or parsed.port not in {None, 443}
            ):
                raise ValueError("capture requires an official unauthenticated HTTPS API")
            if parsed.query or parsed.fragment:
                raise ValueError("capture query parameters must be declared separately")
            path = parsed.path.rstrip("/")
            if host == "api.open-meteo.com":
                if role != "forecast" or path != "/v1/forecast":
                    raise ValueError("only prospective forecast weather can be captured")
            elif host == "api.openf1.org":
                allowed = {
                    "session_metadata": "/v1/sessions",
                    "session_laps": "/v1/laps",
                    "session_stints": "/v1/stints",
                    "session_pits": "/v1/pit",
                    "roster": "/v1/drivers",
                }
                if allowed.get(role) != path:
                    raise ValueError("unsupported prospective OpenF1 endpoint")
            else:
                allowed = {
                    "event": f"/ergast/f1/{event.season}",
                    "qualifying": f"/ergast/f1/{event.season}/{event.round}/qualifying",
                }
                if allowed.get(role) != path:
                    raise ValueError("only schedule and qualifying Jolpica captures are supported")
            params = item.get("params", {})
            session_key = None
            if host == "api.openf1.org":
                raw_key = params.get("session_key")
                if type(raw_key) not in {int, str} or not re.fullmatch(
                    r"[1-9][0-9]*", str(raw_key)
                ):
                    raise ValueError("prospective OpenF1 requests need an exact session key")
                session_key = int(raw_key)
                if role != "session_metadata" and session_key not in validated_sessions:
                    raise ValueError("capture session metadata before dependent session inputs")
            if host not in clients:
                clients[host] = JsonSourceClient(_LIMITS[host], http_client)
            source = clients[host]
            payload = source.get_json(url, params)
            if host != "api.openf1.org" and not isinstance(payload, dict):
                raise ValueError("prospective response must be a JSON object")
            if host == "api.openf1.org":
                if (
                    not isinstance(payload, list)
                    or not all(isinstance(row, dict) for row in payload)
                    or any(
                        type(row.get("session_key")) is not int or row["session_key"] != session_key
                        for row in payload
                    )
                ):
                    raise ValueError("mixed-session prospective response")
                if role == "session_metadata":
                    if len(payload) != 1 or payload[0].get("session_name") not in {
                        "Practice 1",
                        "Practice 2",
                        "Practice 3",
                        "Qualifying",
                    }:
                        raise ValueError("race sessions cannot be captured as predictive inputs")
                    end = datetime.fromisoformat(payload[0]["date_end"])
                    require_utc(end, "session date_end")
                    if (
                        end > datetime.now(UTC)
                        or end.year != event.season
                        or not 0 <= (race_start.date() - end.date()).days <= 4
                    ):
                        raise ValueError(
                            "captured session is unfinished or belongs to another weekend"
                        )
                    assert session_key is not None
                    validated_sessions.add(session_key)
            elif host == "api.jolpi.ca" and role == "qualifying":
                races = payload["MRData"]["RaceTable"]["Races"]
                if any(
                    int(row["season"]) != event.season or int(row["round"]) != event.round
                    for row in races
                ):
                    raise ValueError("qualifying capture belongs to another event")
            captured = datetime.now(UTC)
            tables[name] = pa.Table.from_pylist(
                [{"payload_json": json.dumps(payload, sort_keys=True, allow_nan=False)}]
            )
            source_requests[name] = {
                "url": url,
                "params": params,
                "role": role,
                "response_captured_at": captured.isoformat(),
            }
        captured_at = datetime.now(UTC)
        PredictionCutoff(kind, captured_at, minutes).validate(race_start, decision)
        metadata = {
            "source_requests": source_requests,
            "window": {
                "race_start": race_start.isoformat(),
                "qualifying_decision_at": decision.isoformat(),
                "pre_race_minutes": minutes,
            },
        }
        return freeze_bundle(
            root / "data" / "raw" / "prospective",
            event,
            captured_at,
            kind.value,
            captured_at,
            tables,
            metadata,
        )
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("malformed capture request or response") from exc
    finally:
        for client in clients.values():
            client.close()
