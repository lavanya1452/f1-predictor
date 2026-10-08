"""Season-aware F1 points allocation, separate from prediction models."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SeasonScoringRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    race_points: list[float] = Field(default_factory=list, min_length=1)
    sprint_points: list[float] = Field(default_factory=list)
    fastest_lap_points: float = Field(default=0.0, ge=0)
    fastest_lap_top_n: int | None = Field(default=None, ge=1)

    @field_validator("race_points", "sprint_points")
    @classmethod
    def non_negative_points(cls, points: list[float]) -> list[float]:
        if any(point < 0 for point in points):
            raise ValueError("Scoring points must be non-negative")
        return points


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seasons: dict[int, SeasonScoringRules] = Field(default_factory=dict)

    @classmethod
    def historical_defaults(cls) -> "ScoringConfig":
        seasons: dict[int, SeasonScoringRules] = {}
        modern = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        classic_2003 = [10, 8, 6, 5, 4, 3, 2, 1]
        classic_1991 = [10, 6, 4, 3, 2, 1]
        classic_1961 = [9, 6, 4, 3, 2, 1]
        classic_1960 = [8, 6, 4, 3, 2, 1]
        classic_1950 = [8, 6, 4, 3, 2]
        for season in range(1950, 2027):
            if season >= 2010:
                race_points = modern
            elif season >= 2003:
                race_points = classic_2003
            elif season >= 1991:
                race_points = classic_1991
            elif season >= 1961:
                race_points = classic_1961
            elif season == 1960:
                race_points = classic_1960
            else:
                race_points = classic_1950
            if season == 2021:
                sprint_points = [3, 2, 1]
            elif season >= 2022:
                sprint_points = [8, 7, 6, 5, 4, 3, 2, 1]
            else:
                sprint_points = []
            fastest_lap = (
                1.0
                if 1950 <= season <= 1959 or 2019 <= season <= 2024
                else 0.0
            )
            seasons[season] = SeasonScoringRules(
                race_points=race_points,
                sprint_points=sprint_points,
                fastest_lap_points=fastest_lap,
                fastest_lap_top_n=10 if 2019 <= season <= 2024 else None,
            )
        return cls(seasons=seasons)


class ScoringEngine:
    """Score classified race or sprint results using selected season rules."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig.historical_defaults()

    @classmethod
    def from_file(cls, path: str | Path) -> "ScoringEngine":
        try:
            raw = Path(path).read_text(encoding="utf-8")
            return cls(ScoringConfig.model_validate(json.loads(raw)))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Could not load scoring configuration from {path}: {error}") from error

    def rules_for_season(self, season: int) -> SeasonScoringRules:
        if season in self.config.seasons:
            return self.config.seasons[season]
        raise ValueError(
            f"No scoring rules configured for season {season}; supply an explicit season rule"
        )

    def score_results(
        self,
        results: list[dict[str, object]],
        season: int,
        race_type: Literal["race", "sprint"] = "race",
        constructor_by_driver: dict[str, str] | None = None,
        fastest_lap_driver_id: str | None = None,
    ) -> dict[str, object]:
        if race_type not in ("race", "sprint"):
            raise ValueError(f"Unsupported race type: {race_type}")
        rules = self.rules_for_season(season)
        position_points = (
            rules.race_points if race_type == "race" else rules.sprint_points
        )
        if not position_points:
            raise ValueError(f"No {race_type} points are configured for season {season}")
        driver_points: dict[str, float] = {}
        constructor_points: dict[str, float] = {}
        seen_drivers: set[str] = set()
        seen_positions: set[int] = set()
        classified: dict[str, int] = {}

        for result in results:
            driver_id = result.get("driver_id")
            if not isinstance(driver_id, str) or not driver_id:
                raise ValueError("Every result must have a non-empty driver_id")
            if driver_id in seen_drivers:
                raise ValueError(f"Duplicate result for driver {driver_id}")
            seen_drivers.add(driver_id)
            position = result.get("position")
            is_classified = result.get("classified")
            if is_classified is None:
                is_classified = str(result.get("status", "")).lower() == "finished"
            points = 0.0
            if is_classified:
                if not isinstance(position, int) or position < 1:
                    raise ValueError(f"Classified result for {driver_id} needs a valid position")
                if position in seen_positions:
                    raise ValueError(f"Duplicate classified position {position}")
                seen_positions.add(position)
                classified[driver_id] = position
                if position <= len(position_points):
                    points += position_points[position - 1]
            driver_points[driver_id] = points

            constructor_id = result.get("constructor_id")
            if constructor_id is None and constructor_by_driver is not None:
                constructor_id = constructor_by_driver.get(driver_id)
            if constructor_id is not None:
                if not isinstance(constructor_id, str) or not constructor_id:
                    raise ValueError(f"Invalid constructor for driver {driver_id}")
                constructor_points[constructor_id] = (
                    constructor_points.get(constructor_id, 0.0) + points
                )

        if race_type == "race" and fastest_lap_driver_id is not None:
            if fastest_lap_driver_id not in seen_drivers:
                raise ValueError("Fastest-lap driver must have a result in this race")
            position = classified.get(fastest_lap_driver_id)
            eligible = position is not None and (
                rules.fastest_lap_top_n is None or position <= rules.fastest_lap_top_n
            )
            if eligible:
                driver_points[fastest_lap_driver_id] += rules.fastest_lap_points
                result = next(
                    row for row in results if row.get("driver_id") == fastest_lap_driver_id
                )
                constructor_id = result.get("constructor_id")
                if constructor_id is None and constructor_by_driver is not None:
                    constructor_id = constructor_by_driver.get(fastest_lap_driver_id)
                if isinstance(constructor_id, str):
                    constructor_points[constructor_id] = (
                        constructor_points.get(constructor_id, 0.0)
                        + rules.fastest_lap_points
                    )

        return {
            "season": season,
            "race_type": race_type,
            "driver_points": driver_points,
            "constructor_points": constructor_points,
        }
