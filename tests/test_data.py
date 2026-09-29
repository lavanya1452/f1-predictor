"""Tests for Stage 1 schemas, configuration, and cached API ingestion."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.data.ingestion import JolpicaClient
from src.data.schemas import Race
from src.utils.config import AppConfig


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, str], timeout: float) -> FakeResponse:
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

    assert [season["season"] for season in seasons] == ["2020", "2021", "2022"]
    assert repeated_seasons == seasons
    assert first_request_count == 2
    assert len(fake_session.calls) == first_request_count
    assert len(list(config.raw_data_dir.glob("*.json"))) == 2