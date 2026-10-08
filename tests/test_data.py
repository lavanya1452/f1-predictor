"""Tests for Stage 1 schemas, configuration, and cached API ingestion."""

from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from src.data.ingestion import DataSourceError, JolpicaClient
from src.data.pipeline import DataValidationError, HistoricalDataPipeline
from src.data.schemas import Race
from src.utils.config import AppConfig


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        params = kwargs["params"]
        timeout = kwargs["timeout"]
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        offset = int(params["offset"])
        data = [
            {"season": "2020"},
            {"season": "2021"},
            {"season": "2022"},
        ]
        page = data[offset : offset + int(params["limit"])]
        return FakeResponse(
            {
                "MRData": {
                    "total": str(len(data)),
                    "offset": str(offset),
                    "SeasonTable": {"Seasons": page},
                }
            }
        )


class StaticSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def get(self, _url: str, **_kwargs: Any) -> FakeResponse:
        _ = (_url, _kwargs)
        self.calls += 1
        return FakeResponse(self.payload)


def test_race_schema_requires_canonical_identifiers() -> None:
    race = Race(
        season=2024,
        round=1,
        race_id="2024_1",
        circuit_id="bahrain",
        name="Bahrain Grand Prix",
        date="2024-03-02",
    )

    assert race.season == 2024
    assert race.date.isoformat() == "2024-03-02"
    with pytest.raises(ValidationError):
        Race(
            season=2024,
            round=1,
            race_id="2024_1",
            circuit_id="bahrain",
            name="Bahrain Grand Prix",
        )


def test_config_resolves_relative_data_paths(tmp_path: Path) -> None:
    config = AppConfig(project_root=tmp_path, raw_data_dir=Path("data/raw"))

    assert config.raw_data_dir == tmp_path / "data" / "raw"


def test_jolpica_client_fetches_pages_and_reuses_cache(tmp_path: Path) -> None:
    fake_session = FakeSession()
    config = AppConfig(
        project_root=tmp_path,
        raw_data_dir=Path("data/raw"),
        api_page_size=2,
    )
    client = JolpicaClient(config=config, session=fake_session)

    seasons = client.fetch_seasons()
    first_request_count = len(fake_session.calls)
    repeated_seasons = client.fetch_seasons()

    assert [season["season"] for season in seasons] == [2020, 2021, 2022]
    assert repeated_seasons == seasons
    assert first_request_count == 2
    assert len(fake_session.calls) == first_request_count
    assert len(list(config.raw_data_dir.glob("*.json"))) == 2


def test_jolpica_maps_nested_race_results_and_standings(tmp_path: Path) -> None:
    response = {
        "MRData": {
            "total": "1",
            "offset": "0",
            "RaceTable": {
                "Races": [
                    {
                        "season": "2024",
                        "round": "1",
                        "QualifyingResults": [
                            {
                                "position": "1",
                                "Driver": {"driverId": "driver-a"},
                                "Constructor": {"constructorId": "team-a"},
                                "Q1": "1:30.000",
                                "Q2": "1:29.000",
                                "Q3": "1:28.000",
                            }
                        ],
                        "Results": [
                            {
                                "position": "R",
                                "positionText": "R",
                                "Driver": {"driverId": "driver-a"},
                                "Constructor": {"constructorId": "team-a"},
                                "grid": "2",
                                "points": "0",
                                "laps": "4",
                                "status": "Engine",
                                "FastestLap": {"rank": "3"},
                            }
                        ],
                    }
                ]
            },
        }
    }
    config = AppConfig(project_root=tmp_path, api_retries=0)
    qualifying = JolpicaClient(config, StaticSession(response)).fetch_qualifying_results(2024, 1)
    race_results = JolpicaClient(config, StaticSession(response)).fetch_race_results(2024, 1)

    assert qualifying == [
        {
            "season": 2024,
            "round": 1,
            "race_id": "2024-1",
            "driver_id": "driver-a",
            "constructor_id": "team-a",
            "position": 1,
            "q1": "1:30.000",
            "q2": "1:29.000",
            "q3": "1:28.000",
        }
    ]
    assert race_results[0]["position"] is None
    assert race_results[0]["position_text"] == "R"
    assert race_results[0]["fastest_lap_rank"] == 3

    standings_response = {
        "MRData": {
            "total": "1",
            "offset": "0",
            "StandingsTable": {
                "StandingsLists": [
                    {
                        "season": "2024",
                        "round": "1",
                        "DriverStandings": [
                            {
                                "position": "1",
                                "points": "26",
                                "wins": "1",
                                "Driver": {"driverId": "driver-a"},
                            }
                        ],
                        "ConstructorStandings": [
                            {
                                "position": "1",
                                "points": "26",
                                "wins": "1",
                                "Constructor": {"constructorId": "team-a"},
                            }
                        ],
                    }
                ]
            },
        }
    }
    driver_standings = JolpicaClient(
        config, StaticSession(standings_response)
    ).fetch_driver_standings(2024, 1)
    assert driver_standings[0]["driver_id"] == "driver-a"
    assert driver_standings[0]["round"] == 1
    constructor_standings = JolpicaClient(
        config, StaticSession(standings_response)
    ).fetch_constructor_standings(2024, 1)
    assert constructor_standings[0]["constructor_id"] == "team-a"


def test_jolpica_maps_catalog_and_race_records(tmp_path: Path) -> None:
    config = AppConfig(project_root=tmp_path, api_retries=0)

    race_payload = {
        "MRData": {
            "total": "1",
            "offset": "0",
            "RaceTable": {
                "Races": [
                    {
                        "season": "2024",
                        "round": "1",
                        "raceName": "Bahrain Grand Prix",
                        "date": "2024-03-02",
                        "Circuit": {"circuitId": "bahrain"},
                    }
                ]
            },
        }
    }
    driver_payload = {
        "MRData": {
            "total": "1",
            "offset": "0",
            "DriverTable": {
                "Drivers": [
                    {
                        "driverId": "driver-a",
                        "givenName": "A",
                        "familyName": "Driver",
                    }
                ]
            },
        }
    }
    constructor_payload = {
        "MRData": {
            "total": "1",
            "offset": "0",
            "ConstructorTable": {
                "Constructors": [
                    {"constructorId": "team-a", "name": "Team A"}
                ]
            },
        }
    }
    circuit_payload = {
        "MRData": {
            "total": "1",
            "offset": "0",
            "CircuitTable": {
                "Circuits": [
                    {
                        "circuitId": "bahrain",
                        "circuitName": "Bahrain International Circuit",
                        "Location": {
                            "locality": "Sakhir",
                            "country": "Bahrain",
                            "lat": "26.0325",
                            "long": "50.5106",
                        },
                    }
                ]
            },
        }
    }

    assert JolpicaClient(config, StaticSession(race_payload)).fetch_races(2024)[0][
        "race_id"
    ] == "2024-1"
    assert JolpicaClient(config, StaticSession(driver_payload)).fetch_drivers()[0][
        "driver_id"
    ] == "driver-a"
    assert JolpicaClient(config, StaticSession(constructor_payload)).fetch_constructors()[
        0
    ]["constructor_id"] == "team-a"
    circuit = JolpicaClient(config, StaticSession(circuit_payload)).fetch_circuits()[0]
    assert circuit["latitude"] == 26.0325
    assert circuit["longitude"] == 50.5106


def test_missing_api_envelope_fails_instead_of_returning_empty_data(tmp_path: Path) -> None:
    client = JolpicaClient(
        AppConfig(project_root=tmp_path, api_retries=0),
        StaticSession({"unexpected": {}}),
    )

    with pytest.raises(DataSourceError, match="Missing MRData"):
        client.fetch_seasons()


def test_inconsistent_pagination_metadata_fails(tmp_path: Path) -> None:
    client = JolpicaClient(
        AppConfig(project_root=tmp_path, api_retries=0),
        StaticSession(
            {
                "MRData": {
                    "total": "3",
                    "offset": "1",
                    "SeasonTable": {"Seasons": [{"season": "2024"}]},
                }
            }
        ),
    )

    with pytest.raises(DataSourceError, match="Inconsistent pagination"):
        client.fetch_seasons()


def test_connection_failure_retries_then_returns_data(tmp_path: Path) -> None:
    class FlakySession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _url: str, **_kwargs: Any) -> FakeResponse:
            _ = (_url, _kwargs)
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("temporary network failure")
            return FakeResponse(
                {
                    "MRData": {
                        "total": "1",
                        "offset": "0",
                        "SeasonTable": {"Seasons": [{"season": "2024"}]},
                    }
                }
            )

    session = FlakySession()
    client = JolpicaClient(
        AppConfig(project_root=tmp_path, api_retry_backoff_seconds=0), session
    )
    assert client.fetch_seasons()[0]["season"] == 2024
    assert session.calls == 2


def test_rate_limit_retry_honors_retry_after(tmp_path: Path) -> None:
    class RateLimitedSession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _url: str, **_kwargs: Any) -> FakeResponse:
            assert _url.endswith("/seasons")
            assert _kwargs["timeout"] > 0
            self.calls += 1
            response = FakeResponse(
                {
                    "MRData": {
                        "total": "1",
                        "offset": "0",
                        "SeasonTable": {"Seasons": [{"season": "2024"}]},
                    }
                },
                status_code=429 if self.calls == 1 else 200,
            )
            if self.calls == 1:
                response.headers["Retry-After"] = "0"
            return response

    session = RateLimitedSession()
    client = JolpicaClient(
        AppConfig(project_root=tmp_path, api_retry_backoff_seconds=0), session
    )

    assert client.fetch_seasons()[0]["season"] == 2024
    assert session.calls == 2


def test_pipeline_rejects_duplicate_and_impossible_result_data() -> None:
    race = {
        "season": 2024,
        "round": 1,
        "race_id": "2024-1",
        "circuit_id": "circuit-a",
        "name": "Test Grand Prix",
        "date": "2024-03-02",
    }
    datasets = {
        "seasons": [{"season": 2024}],
        "races": [race, race],
    }

    with pytest.raises(DataValidationError, match="Duplicate races"):
        HistoricalDataPipeline.validate_datasets(datasets)

    datasets["races"] = [race]
    datasets["race_results"] = [
        {
            "season": 2024,
            "round": 1,
            "race_id": "2024-1",
            "driver_id": "driver-a",
            "constructor_id": "team-a",
            "grid": 1,
            "position": 2,
            "position_text": "2",
            "points": 18,
            "laps": 57,
            "status": "Finished",
        }
    ]
    with pytest.raises(DataValidationError, match="Impossible finishing position"):
        HistoricalDataPipeline.validate_datasets(datasets)