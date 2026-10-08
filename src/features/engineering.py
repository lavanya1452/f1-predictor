"""Build race-entry features from information available before each race."""

from bisect import bisect_left
from collections import defaultdict
from statistics import mean, median
from typing import Any

import pandas as pd

from src.data.loaders import load_jsonl
from src.data.schemas import ConstructorStanding, DriverStanding, QualifyingResult, Race, RaceResult
from src.utils.config import AppConfig

FEATURE_COLUMNS = [
    "season",
    "round",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "grid_position",
    "qualifying_position",
    "driver_previous_finish",
    "driver_rolling_finish_mean_5",
    "driver_rolling_finish_median_5",
    "driver_recent_finish_trend",
    "driver_previous_qualifying_position",
    "driver_rolling_qualifying_mean_5",
    "driver_points_entering_race",
    "driver_championship_position_entering_race",
    "driver_dnf_rate",
    "driver_recent_dnf_rate_5",
    "driver_starts_before_race",
    "driver_completed_races_before_race",
    "driver_circuit_finish_mean",
    "driver_circuit_starts",
    "constructor_previous_finish",
    "constructor_rolling_finish_mean_5",
    "constructor_points_entering_race",
    "constructor_championship_position_entering_race",
    "constructor_dnf_rate",
]

TARGET_COLUMNS = ["target_position", "target_points", "target_dnf"]
IDENTITY_COLUMNS = ["race_id", "driver_id", "season", "round"]


def _position(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_dnf(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized != "finished" and not (
        normalized.startswith("+") and "lap" in normalized
    )


class FeatureBuilder:
    """Create pre-race rows using only prior results and pre-race grid inputs.

    Grid and qualifying position for the target race are included because this
    matrix is intended for predictions made after qualifying. To predict before
    qualifying, callers must leave those values absent.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.from_env()

    def build_from_processed(self) -> pd.DataFrame:
        data_dir = self.config.processed_data_dir
        return self.build_training_frame(
            races=load_jsonl(data_dir / "races.jsonl"),
            qualifying_results=load_jsonl(data_dir / "qualifying_results.jsonl"),
            race_results=load_jsonl(data_dir / "race_results.jsonl"),
            driver_standings=load_jsonl(data_dir / "driver_standings.jsonl"),
            constructor_standings=load_jsonl(data_dir / "constructor_standings.jsonl"),
        )

    @staticmethod
    def build_training_frame(
        races: list[dict[str, Any]],
        qualifying_results: list[dict[str, Any]],
        race_results: list[dict[str, Any]],
        driver_standings: list[dict[str, Any]] | None = None,
        constructor_standings: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        race_by_id = {
            row["race_id"]: Race.model_validate(row).model_dump(mode="json")
            for row in races
        }
        results = [
            RaceResult.model_validate(row).model_dump(mode="json")
            for row in race_results
        ]
        qualifying = [
            QualifyingResult.model_validate(row).model_dump(mode="json")
            for row in qualifying_results
        ]
        driver_snapshots = [
            DriverStanding.model_validate(row).model_dump(mode="json")
            for row in (driver_standings or [])
        ]
        constructor_snapshots = [
            ConstructorStanding.model_validate(row).model_dump(mode="json")
            for row in (constructor_standings or [])
        ]
        for result in results:
            if result["race_id"] not in race_by_id:
                raise ValueError(f"Missing race metadata for {result['race_id']}")

        results_by_event: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            results_by_event[(result["season"], result["round"])].append(result)
        qualifying_by_event_driver = {
            (row["season"], row["round"], row["driver_id"]): row["position"]
            for row in qualifying
        }
        qualifying_by_event: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in qualifying:
            qualifying_by_event[(row["season"], row["round"])].append(row)
        driver_standings_by_event = FeatureBuilder._standing_snapshots(
            driver_snapshots, "driver_id"
        )
        constructor_standings_by_event = FeatureBuilder._standing_snapshots(
            constructor_snapshots, "constructor_id"
        )
        driver_standing_keys = sorted(driver_standings_by_event)
        constructor_standing_keys = sorted(constructor_standings_by_event)

        driver_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        constructor_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        circuit_driver_history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        qualifying_history: dict[str, list[int]] = defaultdict(list)
        rows: list[dict[str, Any]] = []

        for event in sorted(results_by_event):
            event_results = results_by_event[event]
            for result in event_results:
                race = race_by_id[result["race_id"]]
                circuit_id = race["circuit_id"]
                driver_id = result["driver_id"]
                constructor_id = result["constructor_id"]
                prior_driver = driver_history[driver_id]
                prior_constructor = constructor_history[constructor_id]
                prior_circuit = circuit_driver_history[(driver_id, circuit_id)]
                prior_driver_positions = [
                    _position(item["position"])
                    for item in prior_driver
                    if item["position"] is not None
                ]
                prior_constructor_positions = [
                    _position(item["position"])
                    for item in prior_constructor
                    if item["position"] is not None
                ]
                recent_positions = prior_driver_positions[-5:]
                previous_three = prior_driver_positions[-6:-3]
                latest_standing_event = FeatureBuilder._latest_prior_event(
                    driver_standing_keys, event
                )
                driver_standing = (
                    driver_standings_by_event[latest_standing_event].get(driver_id, {})
                    if latest_standing_event is not None
                    else {}
                )
                latest_constructor_event = FeatureBuilder._latest_prior_event(
                    constructor_standing_keys, event
                )
                constructor_standing = (
                    constructor_standings_by_event[latest_constructor_event].get(
                        constructor_id, {}
                    )
                    if latest_constructor_event is not None
                    else {}
                )
                qualifying_positions = qualifying_history[driver_id]
                circuit_positions = [
                    _position(item["position"])
                    for item in prior_circuit
                    if item["position"] is not None
                ]
                driver_recent_events = prior_driver[-5:]
                row = {
                    "race_id": result["race_id"],
                    "season": result["season"],
                    "round": result["round"],
                    "driver_id": driver_id,
                    "constructor_id": constructor_id,
                    "circuit_id": circuit_id,
                    "grid_position": _position(result["grid"]),
                    "qualifying_position": qualifying_by_event_driver.get(
                        (result["season"], result["round"], driver_id)
                    ),
                    "driver_previous_finish": (
                        _position(prior_driver[-1]["position"]) if prior_driver else None
                    ),
                    "driver_rolling_finish_mean_5": (
                        mean(recent_positions) if recent_positions else None
                    ),
                    "driver_rolling_finish_median_5": (
                        median(recent_positions) if recent_positions else None
                    ),
                    "driver_recent_finish_trend": (
                        mean(recent_positions[-3:]) - mean(previous_three)
                        if len(recent_positions) >= 3 and len(previous_three) == 3
                        else None
                    ),
                    "driver_previous_qualifying_position": (
                        qualifying_positions[-1] if qualifying_positions else None
                    ),
                    "driver_rolling_qualifying_mean_5": (
                        mean(qualifying_positions[-5:]) if qualifying_positions else None
                    ),
                    "driver_points_entering_race": driver_standing.get("points"),
                    "driver_championship_position_entering_race": driver_standing.get(
                        "position"
                    ),
                    "driver_dnf_rate": FeatureBuilder._dnf_rate(prior_driver),
                    "driver_recent_dnf_rate_5": FeatureBuilder._dnf_rate(
                        driver_recent_events
                    ),
                    "driver_starts_before_race": len(prior_driver),
                    "driver_completed_races_before_race": sum(
                        item["position"] is not None for item in prior_driver
                    ),
                    "driver_circuit_finish_mean": (
                        mean(circuit_positions) if circuit_positions else None
                    ),
                    "driver_circuit_starts": len(prior_circuit),
                    "constructor_previous_finish": (
                        FeatureBuilder._latest_constructor_finish(prior_constructor)
                        if prior_constructor
                        else None
                    ),
                    "constructor_rolling_finish_mean_5": (
                        mean(prior_constructor_positions[-5:])
                        if prior_constructor_positions
                        else None
                    ),
                    "constructor_points_entering_race": constructor_standing.get("points"),
                    "constructor_championship_position_entering_race": (
                        constructor_standing.get("position")
                    ),
                    "constructor_dnf_rate": FeatureBuilder._dnf_rate(prior_constructor),
                    "target_position": _position(result["position"]),
                    "target_points": result["points"],
                    "target_dnf": int(_is_dnf(result["status"])),
                }
                rows.append(row)

            for result in event_results:
                race = race_by_id[result["race_id"]]
                driver_history[result["driver_id"]].append(result)
                constructor_history[result["constructor_id"]].append(result)
                circuit_driver_history[
                    (result["driver_id"], race["circuit_id"])
                ].append(result)
            for qualifying_result in qualifying_by_event.get(event, []):
                qualifying_history[qualifying_result["driver_id"]].append(
                    qualifying_result["position"]
                )

        columns = list(dict.fromkeys(IDENTITY_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS))
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _standing_snapshots(
        records: list[dict[str, Any]], identity_field: str
    ) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
        snapshots: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for record in records:
            snapshots[(record["season"], record["round"])][
                record[identity_field]
            ] = record
        return dict(snapshots)

    @staticmethod
    def _latest_prior_event(
        available_events: list[tuple[int, int]], target: tuple[int, int]
    ) -> tuple[int, int] | None:
        index = bisect_left(available_events, target) - 1
        return available_events[index] if index >= 0 else None

    @staticmethod
    def _dnf_rate(records: list[dict[str, Any]]) -> float | None:
        if not records:
            return None
        return mean(_is_dnf(record["status"]) for record in records)

    @staticmethod
    def _latest_constructor_finish(records: list[dict[str, Any]]) -> float | None:
        latest_event = (records[-1]["season"], records[-1]["round"])
        positions = [
            _position(record["position"])
            for record in records
            if (record["season"], record["round"]) == latest_event
            and record["position"] is not None
        ]
        return mean(positions) if positions else None
