"""Cached, paginated client for the public Jolpica/Ergast-compatible API."""

import hashlib
import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from src.data.schemas import (
    CanonicalRecord,
    Circuit,
    Constructor,
    ConstructorStanding,
    Driver,
    DriverStanding,
    QualifyingResult,
    Race,
    RaceResult,
    Season,
)
from src.utils.config import AppConfig
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class DataSourceError(RuntimeError):
    """Raised when the configured F1 data provider cannot return valid data."""


class JolpicaClient:
    """Retrieve raw Jolpica API responses and cache them locally as JSON.

    API payloads remain raw at this stage; schema mapping is deliberately
    separate so source data is not silently altered during ingestion.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        session: requests.Session | None = None,
        refresh_cache: bool = False,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.session = session or requests.Session()
        self.refresh_cache = refresh_cache

    def _cache_path(self, endpoint: str, params: dict[str, str]) -> Path:
        request_key = json.dumps(
            {
                "base_url": self.config.api_base_url.rstrip("/"),
                "endpoint": endpoint.strip("/"),
                "params": params,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()[:16]
        safe_endpoint = endpoint.strip("/").replace("/", "_") or "root"
        return self.config.raw_data_dir / f"{safe_endpoint}_{digest}.json"

    def _get_json(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        cache_path = self._cache_path(endpoint, params)
        if cache_path.is_file() and not self.refresh_cache:
            try:
                with cache_path.open(encoding="utf-8") as cache_file:
                    payload = json.load(cache_file)
            except (OSError, json.JSONDecodeError) as error:
                raise DataSourceError(f"Could not read API cache {cache_path}: {error}") from error
            if not isinstance(payload, dict):
                raise DataSourceError(f"Cached API response is not a JSON object: {cache_path}")
            return payload

        url = urljoin(f"{self.config.api_base_url.rstrip('/')}/", endpoint.lstrip("/"))
        for attempt in range(self.config.api_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.api_timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.config.api_retries:
                        time.sleep(self._retry_delay(response, attempt))
                        continue
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException as error:
                if attempt >= self.config.api_retries:
                    raise DataSourceError(f"Jolpica request failed for {url}: {error}") from error
                time.sleep(self.config.api_retry_backoff_seconds * (2**attempt))
            except ValueError as error:
                raise DataSourceError(f"Jolpica returned invalid JSON for {url}: {error}") from error
        else:
            raise DataSourceError(f"Jolpica request exhausted retries for {url}")

        if not isinstance(payload, dict):
            raise DataSourceError(f"Jolpica returned a non-object JSON response for {url}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file, ensure_ascii=True, indent=2)
            temporary_path.replace(cache_path)
        except OSError as error:
            raise DataSourceError(f"Could not cache Jolpica response at {cache_path}: {error}") from error
        return payload

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    LOGGER.warning("Could not parse Retry-After header %r", retry_after)
        return self.config.api_retry_backoff_seconds * (2**attempt)

    def fetch_table(
        self,
        endpoint: str,
        table_key: str,
        collection_key: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and concatenate one API table, following its offset metadata."""
        records: list[dict[str, Any]] = []
        offset = 0
        base_params = dict(params or {})

        for _ in range(self.config.max_api_pages):
            page_params = {
                **base_params,
                "limit": str(self.config.api_page_size),
                "offset": str(offset),
                "format": "json",
            }
            payload = self._get_json(endpoint, page_params)
            mr_data = payload.get("MRData")
            if not isinstance(mr_data, dict):
                raise DataSourceError(f"Missing MRData object in Jolpica response for {endpoint}")
            table = mr_data.get(table_key)
            if not isinstance(table, dict):
                raise DataSourceError(f"Missing {table_key} object in Jolpica response for {endpoint}")
            if collection_key not in table:
                raise DataSourceError(
                    f"Missing {collection_key} collection in Jolpica response for {endpoint}"
                )
            page_records = table[collection_key]
            if not isinstance(page_records, list):
                raise DataSourceError(
                    f"Unexpected {collection_key} structure in Jolpica response for {endpoint}"
                )
            if not all(isinstance(record, dict) for record in page_records):
                raise DataSourceError(
                    f"Jolpica returned a non-object record in {collection_key} for {endpoint}"
                )
            records.extend(page_records)

            try:
                total = int(mr_data.get("total", len(page_records)))
                response_offset = int(mr_data.get("offset", offset))
            except (AttributeError, TypeError, ValueError) as error:
                raise DataSourceError(f"Invalid pagination metadata for {endpoint}") from error
            if (
                total < 0
                or response_offset < 0
                or response_offset != offset
                or response_offset + len(page_records) > total
            ):
                raise DataSourceError(f"Inconsistent pagination metadata for {endpoint}")
            if not page_records and response_offset < total:
                raise DataSourceError(f"Jolpica returned an empty page before total for {endpoint}")
            if response_offset + len(page_records) >= total:
                return records
            offset = response_offset + len(page_records)

        raise DataSourceError(
            f"Jolpica pagination exceeded {self.config.max_api_pages} pages for {endpoint}"
        )

    def fetch_seasons(self) -> list[dict[str, Any]]:
        records = self.fetch_table("seasons", "SeasonTable", "Seasons")
        return [
            Season.model_validate(
                {"season": row.get("season"), "url": row.get("url")}
            ).model_dump(mode="json")
            for row in records
        ]

    def fetch_races(self, season: int) -> list[dict[str, Any]]:
        races = self.fetch_table(f"{season}/races", "RaceTable", "Races")
        return [self._canonical_race(race) for race in races]

    @staticmethod
    def _canonical_race(record: dict[str, Any]) -> dict[str, Any]:
        circuit = record.get("Circuit")
        if not isinstance(circuit, dict) or not isinstance(circuit.get("circuitId"), str):
            raise DataSourceError("Race response is missing Circuit.circuitId")
        try:
            season = int(record["season"])
            round_number = int(record["round"])
        except (KeyError, TypeError, ValueError) as error:
            raise DataSourceError("Race response has invalid season or round") from error
        return Race.model_validate(
            {
                "season": season,
                "round": round_number,
                "race_id": f"{season}-{round_number}",
                "circuit_id": circuit["circuitId"],
                "name": record.get("raceName"),
                "date": record.get("date"),
                "time": record.get("time"),
                "url": record.get("url"),
            }
        ).model_dump(mode="json")

    def fetch_drivers(self, season: int | None = None) -> list[dict[str, Any]]:
        endpoint = f"{season}/drivers" if season is not None else "drivers"
        records = self.fetch_table(endpoint, "DriverTable", "Drivers")
        return [
            Driver.model_validate(
                {
                    "driver_id": row.get("driverId"),
                    "permanent_number": row.get("permanentNumber"),
                    "code": row.get("code"),
                    "given_name": row.get("givenName"),
                    "family_name": row.get("familyName"),
                    "date_of_birth": row.get("dateOfBirth"),
                    "nationality": row.get("nationality"),
                }
            ).model_dump(mode="json")
            for row in records
        ]

    def fetch_constructors(self, season: int | None = None) -> list[dict[str, Any]]:
        endpoint = f"{season}/constructors" if season is not None else "constructors"
        records = self.fetch_table(endpoint, "ConstructorTable", "Constructors")
        return [
            Constructor.model_validate(
                {
                    "constructor_id": row.get("constructorId"),
                    "name": row.get("name"),
                    "nationality": row.get("nationality"),
                }
            ).model_dump(mode="json")
            for row in records
        ]

    def fetch_circuits(self, season: int | None = None) -> list[dict[str, Any]]:
        endpoint = f"{season}/circuits" if season is not None else "circuits"
        records = self.fetch_table(endpoint, "CircuitTable", "Circuits")
        normalized: list[dict[str, Any]] = []
        for row in records:
            location = row.get("Location")
            if not isinstance(location, dict):
                location = {}
            normalized.append(
                {
                    "circuit_id": row.get("circuitId"),
                    "name": row.get("circuitName"),
                    "locality": location.get("locality"),
                    "country": location.get("country"),
                    "latitude": location.get("lat"),
                    "longitude": location.get("long"),
                }
            )
        return [Circuit.model_validate(row).model_dump(mode="json") for row in normalized]

    @staticmethod
    def _race_children(
        records: list[dict[str, Any]],
        child_key: str,
        result_model: type[QualifyingResult] | type[RaceResult],
    ) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for race in records:
            try:
                season = int(race["season"])
                round_number = int(race["round"])
                children = race[child_key]
            except (KeyError, TypeError, ValueError) as error:
                raise DataSourceError(f"Race response is missing required {child_key} data") from error
            if not isinstance(children, list):
                raise DataSourceError(f"Expected {child_key} to be a list")
            for result in children:
                if not isinstance(result, dict):
                    raise DataSourceError(f"Expected object in {child_key}")
                driver = result.get("Driver")
                constructor = result.get("Constructor")
                if not isinstance(driver, dict) or not isinstance(constructor, dict):
                    raise DataSourceError(f"{child_key} record lacks driver or constructor")
                common = {
                    "season": season,
                    "round": round_number,
                    "race_id": f"{season}-{round_number}",
                    "driver_id": driver.get("driverId"),
                    "constructor_id": constructor.get("constructorId"),
                    "position": result.get("position"),
                }
                if result_model is QualifyingResult:
                    source = {
                        **common,
                        "q1": result.get("Q1"),
                        "q2": result.get("Q2"),
                        "q3": result.get("Q3"),
                    }
                else:
                    position_text = result.get("positionText", result.get("position", ""))
                    try:
                        position = int(result.get("position"))
                    except (TypeError, ValueError):
                        position = None
                    fastest_lap = result.get("FastestLap")
                    source = {
                        **common,
                        "position": position,
                        "position_text": str(position_text),
                        "grid": result.get("grid"),
                        "points": result.get("points"),
                        "laps": result.get("laps"),
                        "status": result.get("status"),
                        "fastest_lap_rank": (
                            fastest_lap.get("rank") if isinstance(fastest_lap, dict) else None
                        ),
                    }
                flattened.append(result_model.model_validate(source).model_dump(mode="json"))
        return flattened

    def fetch_qualifying_results(self, season: int, round_number: int) -> list[dict[str, Any]]:
        endpoint = f"{season}/{round_number}/qualifying"
        races = self.fetch_table(endpoint, "RaceTable", "Races")
        return self._race_children(races, "QualifyingResults", QualifyingResult)

    def fetch_race_results(self, season: int, round_number: int) -> list[dict[str, Any]]:
        endpoint = f"{season}/{round_number}/results"
        races = self.fetch_table(endpoint, "RaceTable", "Races")
        return self._race_children(races, "Results", RaceResult)

    @staticmethod
    def _flatten_standings(
        records: list[dict[str, Any]],
        entity_key: str,
        entity_id_key: str,
        model: type[DriverStanding] | type[ConstructorStanding],
    ) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for standings_list in records:
            try:
                season = int(standings_list["season"])
                round_number = int(standings_list["round"])
                entries = standings_list[entity_key]
            except (KeyError, TypeError, ValueError) as error:
                raise DataSourceError(f"Standings response is missing {entity_key}") from error
            if not isinstance(entries, list):
                raise DataSourceError(f"Expected {entity_key} to be a list")
            for entry in entries:
                entity = entry.get("Driver" if entity_id_key == "driverId" else "Constructor")
                if not isinstance(entity, dict):
                    raise DataSourceError(f"{entity_key} entry is missing its entity object")
                flattened.append(
                    model.model_validate(
                        {
                            "season": season,
                            "round": round_number,
                            entity_id_key.replace("Id", "_id"): entity.get(entity_id_key),
                            "position": entry.get("position"),
                            "points": entry.get("points"),
                            "wins": entry.get("wins"),
                        }
                    ).model_dump(mode="json")
                )
        return flattened

    def fetch_driver_standings(
        self, season: int, round_number: int | None = None
    ) -> list[dict[str, Any]]:
        endpoint = f"{season}/driverStandings"
        if round_number is not None:
            endpoint = f"{season}/{round_number}/driverStandings"
        records = self.fetch_table(endpoint, "StandingsTable", "StandingsLists")
        return self._flatten_standings(
            records, "DriverStandings", "driverId", DriverStanding
        )

    def fetch_constructor_standings(
        self, season: int, round_number: int | None = None
    ) -> list[dict[str, Any]]:
        endpoint = f"{season}/constructorStandings"
        if round_number is not None:
            endpoint = f"{season}/{round_number}/constructorStandings"
        records = self.fetch_table(endpoint, "StandingsTable", "StandingsLists")
        return self._flatten_standings(
            records, "ConstructorStandings", "constructorId", ConstructorStanding
        )


def validate_records(
    records: list[dict[str, Any]], record_type: type[CanonicalRecord]
) -> list[CanonicalRecord]:
    """Validate dictionaries against a canonical schema without mutating input."""
    return [record_type.model_validate(record) for record in records]