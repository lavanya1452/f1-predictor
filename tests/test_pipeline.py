import json
from pathlib import Path
from typing import Any

import pytest

from src.data.loaders import load_jsonl
from src.data.pipeline import DataValidationError, HistoricalDataPipeline
from src.utils.config import AppConfig


class FixtureProvider:
    def fetch_seasons(self) -> list[dict[str, Any]]:
        return [{"season": 2024, "url": None}]

    def fetch_races(self, season: int) -> list[dict[str, Any]]:
        return [
            {
                "season": season,
                "round": 1,
                "race_id": f"{season}-1",
                "circuit_id": "circuit-a",
                "name": "Fixture Grand Prix",
                "date": "2024-03-02",
            }
        ]

    def fetch_circuits(self, season: int | None = None) -> list[dict[str, Any]]:
        assert season in (None, 2024)
        return [
            {
                "circuit_id": "circuit-a",
                "name": "Fixture Circuit",
                "locality": "Test",
                "country": "Testland",
                "latitude": None,
                "longitude": None,
            }
        ]

    def fetch_drivers(self, season: int | None = None) -> list[dict[str, Any]]:
        assert season in (None, 2024)
        return [
            {
                "driver_id": "driver-a",
                "given_name": "A",
                "family_name": "Driver",
            }
        ]

    def fetch_constructors(self, season: int | None = None) -> list[dict[str, Any]]:
        assert season in (None, 2024)
        return [
            {
                "constructor_id": "team-a",
                "name": "Team A",
            }
        ]

    def fetch_qualifying_results(
        self, season: int, round_number: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "season": season,
                "round": round_number,
                "race_id": f"{season}-{round_number}",
                "driver_id": "driver-a",
                "constructor_id": "team-a",
                "position": 1,
            }
        ]

    def fetch_race_results(
        self, season: int, round_number: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "season": season,
                "round": round_number,
                "race_id": f"{season}-{round_number}",
                "driver_id": "driver-a",
                "constructor_id": "team-a",
                "grid": 1,
                "position": 1,
                "position_text": "1",
                "points": 25,
                "laps": 57,
                "status": "Finished",
            }
        ]

    def fetch_driver_standings(
        self, season: int, round_number: int | None = None
    ) -> list[dict[str, Any]]:
        assert round_number is not None
        return [
            {
                "season": season,
                "round": round_number,
                "driver_id": "driver-a",
                "position": 1,
                "points": 25,
                "wins": 1,
            }
        ]

    def fetch_constructor_standings(
        self, season: int, round_number: int | None = None
    ) -> list[dict[str, Any]]:
        assert round_number is not None
        return [
            {
                "season": season,
                "round": round_number,
                "constructor_id": "team-a",
                "position": 1,
                "points": 25,
                "wins": 1,
            }
        ]


def test_pipeline_downloads_validates_and_persists_processed_data(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        project_root=tmp_path,
        raw_data_dir=Path("data/raw"),
        processed_data_dir=Path("data/processed"),
    )
    datasets = HistoricalDataPipeline(config, FixtureProvider()).download(2024)

    assert len(datasets["race_results"]) == 1
    assert load_jsonl(config.processed_data_dir / "race_results.jsonl")[0]["points"] == 25.0
    manifest = json.loads(
        (config.processed_data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["record_counts"]["races"] == 1


def test_pipeline_rejects_duplicate_identifiers_within_season(tmp_path: Path) -> None:
    class DuplicateDriverProvider(FixtureProvider):
        def fetch_drivers(self, season: int | None = None) -> list[dict[str, Any]]:
            records = super().fetch_drivers(season)
            return records + records

    with pytest.raises(DataValidationError, match="Duplicate drivers"):
        HistoricalDataPipeline(
            AppConfig(project_root=tmp_path), DuplicateDriverProvider()
        ).download(2024)
