"""Cached, paginated client for the public Jolpica/Ergast-compatible API."""

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from src.data.schemas import CanonicalRecord
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
            {"endpoint": endpoint.strip("/"), "params": params},
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
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.api_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise DataSourceError(f"Jolpica request failed for {url}: {error}") from error

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
            mr_data = payload.get("MRData", {})
            table = mr_data.get(table_key, {}) if isinstance(mr_data, dict) else {}
            page_records = table.get(collection_key, []) if isinstance(table, dict) else []
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
            if not page_records or response_offset + len(page_records) >= total:
                return records
            offset = response_offset + len(page_records)

        raise DataSourceError(
            f"Jolpica pagination exceeded {self.config.max_api_pages} pages for {endpoint}"
        )

    def fetch_seasons(self) -> list[dict[str, Any]]:
        return self.fetch_table("seasons", "SeasonTable", "Seasons")

    def fetch_races(self, season: int) -> list[dict[str, Any]]:
        return self.fetch_table(f"{season}/races", "RaceTable", "Races")

    def fetch_drivers(self, season: int | None = None) -> list[dict[str, Any]]:
        endpoint = f"{season}/drivers" if season is not None else "drivers"
        return self.fetch_table(endpoint, "DriverTable", "Drivers")

    def fetch_constructors(self, season: int | None = None) -> list[dict[str, Any]]:
        endpoint = f"{season}/constructors" if season is not None else "constructors"
        return self.fetch_table(endpoint, "ConstructorTable", "Constructors")

    def fetch_circuits(self, season: int | None = None) -> list[dict[str, Any]]:
        endpoint = f"{season}/circuits" if season is not None else "circuits"
        return self.fetch_table(endpoint, "CircuitTable", "Circuits")

    def fetch_qualifying_results(self, season: int, round_number: int) -> list[dict[str, Any]]:
        endpoint = f"{season}/{round_number}/qualifying"
        return self.fetch_table(endpoint, "RaceTable", "Races")

    def fetch_race_results(self, season: int, round_number: int) -> list[dict[str, Any]]:
        endpoint = f"{season}/{round_number}/results"
        return self.fetch_table(endpoint, "RaceTable", "Races")

    def fetch_driver_standings(self, season: int) -> list[dict[str, Any]]:
        return self.fetch_table(
            f"{season}/driverStandings",
            "StandingsTable",
            "StandingsLists",
        )

    def fetch_constructor_standings(self, season: int) -> list[dict[str, Any]]:
        return self.fetch_table(
            f"{season}/constructorStandings",
            "StandingsTable",
            "StandingsLists",
        )


def validate_records(
    records: list[dict[str, Any]], record_type: type[CanonicalRecord]
) -> list[CanonicalRecord]:
    """Validate dictionaries against a canonical schema without mutating input."""
    return [record_type.model_validate(record) for record in records]