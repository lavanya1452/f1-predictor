"""Historical data orchestration and canonical dataset persistence."""

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import ValidationError

from src.data.ingestion import DataSourceError, JolpicaClient
from src.data.schemas import (
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

LOGGER = logging.getLogger(__name__)


class F1DataProvider(Protocol):
    """Provider contract consumed by the historical pipeline."""

    def fetch_seasons(self) -> list[dict[str, Any]]: ...

    def fetch_races(self, season: int) -> list[dict[str, Any]]: ...

    def fetch_circuits(self, season: int | None = None) -> list[dict[str, Any]]: ...

    def fetch_drivers(self, season: int | None = None) -> list[dict[str, Any]]: ...

    def fetch_constructors(self, season: int | None = None) -> list[dict[str, Any]]: ...

    def fetch_qualifying_results(self, season: int, round_number: int) -> list[dict[str, Any]]: ...

    def fetch_race_results(self, season: int, round_number: int) -> list[dict[str, Any]]: ...

    def fetch_driver_standings(
        self, season: int, round_number: int | None = None
    ) -> list[dict[str, Any]]: ...

    def fetch_constructor_standings(
        self, season: int, round_number: int | None = None
    ) -> list[dict[str, Any]]: ...


RECORD_MODELS: dict[str, type[Any]] = {
    "seasons": Season,
    "races": Race,
    "circuits": Circuit,
    "drivers": Driver,
    "constructors": Constructor,
    "qualifying_results": QualifyingResult,
    "race_results": RaceResult,
    "driver_standings": DriverStanding,
    "constructor_standings": ConstructorStanding,
}

UNIQUE_KEYS: dict[str, tuple[str, ...]] = {
    "seasons": ("season",),
    "races": ("race_id",),
    "circuits": ("circuit_id",),
    "drivers": ("driver_id",),
    "constructors": ("constructor_id",),
    "qualifying_results": ("season", "round", "driver_id"),
    "race_results": ("season", "round", "driver_id"),
    "driver_standings": ("season", "round", "driver_id"),
    "constructor_standings": ("season", "round", "constructor_id"),
}


class DataValidationError(ValueError):
    """Raised when downloaded datasets violate canonical data constraints."""


class HistoricalDataPipeline:
    """Download, validate, and persist real Jolpica-compatible historical data."""

    def __init__(self, config: AppConfig | None = None, provider: F1DataProvider | None = None) -> None:
        self.config = config or AppConfig.from_env()
        self.provider = provider or JolpicaClient(config=self.config)

    def available_seasons(self) -> list[int]:
        try:
            seasons = [int(row["season"]) for row in self.provider.fetch_seasons()]
        except (KeyError, TypeError, ValueError) as error:
            raise DataValidationError("Provider returned an invalid season list") from error
        if not seasons:
            raise DataSourceError("The provider returned no available seasons")
        return sorted(set(seasons))

    def download(
        self, start_season: int, end_season: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            season_records = self.provider.fetch_seasons()
            available = sorted({int(row["season"]) for row in season_records})
        except (KeyError, TypeError, ValueError) as error:
            raise DataValidationError("Provider returned an invalid season list") from error
        if not available:
            raise DataSourceError("The provider returned no available seasons")
        requested_end = max(available) if end_season is None else end_season
        final_season = min(requested_end, max(available))
        if start_season < 1950 or final_season < start_season:
            raise ValueError("Season range must satisfy 1950 <= start-season <= end-season")
        seasons = [
            season
            for season in available
            if start_season <= season <= final_season
        ]
        if not seasons:
            raise ValueError(f"No provider seasons available between {start_season} and {final_season}")

        datasets: dict[str, list[dict[str, Any]]] = {
            name: [] for name in RECORD_MODELS
        }
        datasets["seasons"] = [
            row
            for row in season_records
            if start_season <= int(row["season"]) <= final_season
        ]

        for season in seasons:
            LOGGER.info("Downloading F1 season %s", season)
            for name, key, fetch in (
                ("circuits", "circuit_id", self.provider.fetch_circuits),
                ("drivers", "driver_id", self.provider.fetch_drivers),
                ("constructors", "constructor_id", self.provider.fetch_constructors),
            ):
                entity_records = fetch(season)
                entity_ids = [record.get(key) for record in entity_records]
                if len(entity_ids) != len(set(entity_ids)):
                    raise DataValidationError(
                        f"Duplicate {name} identifier within season {season}"
                    )
                datasets[name].extend(entity_records)
            races = self.provider.fetch_races(season)
            datasets["races"].extend(races)
            for race in races:
                round_number = int(race["round"])
                datasets["qualifying_results"].extend(
                    self.provider.fetch_qualifying_results(season, round_number)
                )
                datasets["race_results"].extend(
                    self.provider.fetch_race_results(season, round_number)
                )
                datasets["driver_standings"].extend(
                    self.provider.fetch_driver_standings(season, round_number)
                )
                datasets["constructor_standings"].extend(
                    self.provider.fetch_constructor_standings(season, round_number)
                )

        for name, key_field in (
            ("circuits", "circuit_id"),
            ("drivers", "driver_id"),
            ("constructors", "constructor_id"),
        ):
            by_id: dict[str, dict[str, Any]] = {}
            for record in datasets[name]:
                by_id[record[key_field]] = record
            datasets[name] = list(by_id.values())

        canonical = self.validate_datasets(datasets)
        self.persist(canonical, start_season, final_season)
        return canonical

    @staticmethod
    def validate_datasets(
        datasets: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        unknown = set(datasets) - set(RECORD_MODELS)
        if unknown:
            raise DataValidationError(f"Unknown dataset names: {sorted(unknown)}")
        normalized: dict[str, list[dict[str, Any]]] = {}
        for name, model in RECORD_MODELS.items():
            records = datasets.get(name, [])
            if not isinstance(records, list):
                raise DataValidationError(f"{name} must be a list")
            try:
                normalized[name] = [
                    model.model_validate(record).model_dump(mode="json") for record in records
                ]
            except (ValidationError, TypeError) as error:
                raise DataValidationError(f"Invalid record in {name}: {error}") from error

            key_fields = UNIQUE_KEYS[name]
            seen: set[tuple[Any, ...]] = set()
            for record in normalized[name]:
                key = tuple(record[field] for field in key_fields)
                if key in seen:
                    raise DataValidationError(f"Duplicate {name} record for key {key}")
                seen.add(key)

        races = {
            (row["season"], row["round"]): row["race_id"]
            for row in normalized["races"]
        }
        known_seasons = {row["season"] for row in normalized["seasons"]}
        race_event_keys: set[tuple[int, int]] = set()
        for row in normalized["races"]:
            if known_seasons and row["season"] not in known_seasons:
                raise DataValidationError(
                    f"Race references unknown season {row['season']}"
                )
            event_key = (row["season"], row["round"])
            if event_key in race_event_keys:
                raise DataValidationError(
                    f"Duplicate race record for season/round {event_key}"
                )
            race_event_keys.add(event_key)
        race_ids = {row["race_id"] for row in normalized["races"]}
        for table in ("qualifying_results", "race_results"):
            for row in normalized[table]:
                if (row["season"], row["round"]) not in races or row["race_id"] not in race_ids:
                    raise DataValidationError(f"{table} references an unknown race: {row['race_id']}")
                if row["race_id"] != races[(row["season"], row["round"])]:
                    raise DataValidationError(f"{table} has inconsistent season/round identifiers")
        for table in ("driver_standings", "constructor_standings"):
            for row in normalized[table]:
                if (row["season"], row["round"]) not in races:
                    raise DataValidationError(
                        f"{table} references unknown season/round "
                        f"{row['season']}/{row['round']}"
                    )
        race_result_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in normalized["race_results"]:
            race_result_groups.setdefault((row["season"], row["round"]), []).append(row)
        for (season, round_number), records in race_result_groups.items():
            classified_positions = [
                row["position"] for row in records if row["position"] is not None
            ]
            if len(classified_positions) != len(set(classified_positions)):
                raise DataValidationError(
                    f"Duplicate classified finishing position in {season} round {round_number}"
                )
            for row in records:
                if row["position"] is not None and row["position"] > len(records):
                    raise DataValidationError(
                        f"Impossible finishing position in race {row['race_id']}"
                    )
        available_drivers = {row["driver_id"] for row in normalized["drivers"]}
        available_constructors = {
            row["constructor_id"] for row in normalized["constructors"]
        }
        available_circuits = {row["circuit_id"] for row in normalized["circuits"]}
        for row in normalized["races"]:
            if available_circuits and row["circuit_id"] not in available_circuits:
                raise DataValidationError(f"Race references unknown circuit {row['circuit_id']}")
        for table in ("qualifying_results", "race_results", "driver_standings"):
            for row in normalized[table]:
                if available_drivers and row["driver_id"] not in available_drivers:
                    raise DataValidationError(f"{table} references unknown driver {row['driver_id']}")
        for table in ("qualifying_results", "race_results", "constructor_standings"):
            for row in normalized[table]:
                if available_constructors and row["constructor_id"] not in available_constructors:
                    raise DataValidationError(
                        f"{table} references unknown constructor {row['constructor_id']}"
                    )
        return normalized

    def persist(
        self,
        datasets: dict[str, list[dict[str, Any]]],
        start_season: int,
        end_season: int,
    ) -> None:
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        for name, records in datasets.items():
            path = self.config.processed_data_dir / f"{name}.jsonl"
            temporary_path = path.with_suffix(".tmp")
            try:
                with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
                    for record in records:
                        file.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                temporary_path.replace(path)
            except OSError as error:
                raise DataSourceError(f"Could not write processed dataset {path}: {error}") from error

        manifest = {
            "source": self.config.api_base_url,
            "start_season": start_season,
            "end_season": end_season,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "record_counts": {name: len(records) for name, records in datasets.items()},
        }
        manifest_path = self.config.processed_data_dir / "manifest.json"
        temporary_manifest = manifest_path.with_suffix(".tmp")
        try:
            with temporary_manifest.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, sort_keys=True)
            temporary_manifest.replace(manifest_path)
        except OSError as error:
            raise DataSourceError(
                f"Could not write processed data manifest {manifest_path}: {error}"
            ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical F1 data from Jolpica.")
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, default=None)
    args = parser.parse_args()

    pipeline = HistoricalDataPipeline()
    try:
        datasets = pipeline.download(args.start_season, args.end_season)
    except (DataSourceError, DataValidationError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(
        "Downloaded and validated records: "
        + ", ".join(f"{name}={len(records)}" for name, records in datasets.items())
    )


if __name__ == "__main__":
    main()
