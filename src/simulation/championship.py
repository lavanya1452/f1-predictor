"""Monte Carlo championship simulation over remaining race forecasts."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.scoring.engine import ScoringEngine
from src.simulation.race import RaceSimulator


@dataclass(frozen=True)
class RaceForecast:
    """Precomputed per-driver performance assumptions for one remaining event."""

    season: int
    round: int
    strengths: Mapping[str, float]
    dnf_probabilities: Mapping[str, float] = field(default_factory=dict)
    fastest_lap_probabilities: Mapping[str, float] = field(default_factory=dict)
    sprint_strengths: Mapping[str, float] | None = None
    sprint_dnf_probabilities: Mapping[str, float] = field(default_factory=dict)


def _normalized_probability_map(
    probabilities: Mapping[str, float], eligible: set[str], label: str
) -> dict[str, float]:
    unknown = set(probabilities) - eligible
    if unknown:
        raise ValueError(f"{label} has unknown drivers: {sorted(unknown)}")
    values = {key: float(value) for key, value in probabilities.items()}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"{label} must contain finite non-negative values")
    total = sum(values.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"{label} must sum to one, got {total}")
    return values


class ChampionshipSimulator:
    """Simulate complete remaining schedules and summarize final positions."""

    def __init__(
        self,
        scoring_engine: ScoringEngine | None = None,
        race_simulator: RaceSimulator | None = None,
    ) -> None:
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.race_simulator = race_simulator or RaceSimulator()

    def simulate(
        self,
        current_driver_points: Mapping[str, float],
        remaining_races: list[RaceForecast],
        num_simulations: int = 1000,
        random_seed: int = 42,
        constructor_by_driver: Mapping[str, str] | None = None,
        current_constructor_points: Mapping[str, float] | None = None,
        current_driver_wins: Mapping[str, int] | None = None,
        current_constructor_wins: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        if num_simulations < 1:
            raise ValueError("num_simulations must be at least one")
        if not current_driver_points:
            raise ValueError("Current driver standings cannot be empty")
        if any(not np.isfinite(float(points)) or float(points) < 0 for points in current_driver_points.values()):
            raise ValueError("Championship points must be finite and non-negative")
        race_keys = [(race.season, race.round) for race in remaining_races]
        if race_keys != sorted(race_keys) or len(set(race_keys)) != len(race_keys):
            raise ValueError("Remaining races must be uniquely ordered by season and round")
        drivers = set(current_driver_points)
        for race in remaining_races:
            if not race.strengths:
                raise ValueError(f"Race {race.season}-{race.round} has no eligible drivers")
            drivers.update(race.strengths)
            if race.fastest_lap_probabilities:
                _normalized_probability_map(
                    race.fastest_lap_probabilities,
                    set(race.strengths),
                    "fastest_lap_probabilities",
                )
        constructors = set(constructor_by_driver.values()) if constructor_by_driver else set()
        initial_constructor_points = {
            team: float(points)
            for team, points in (current_constructor_points or {}).items()
        }
        if any(
            not np.isfinite(points) or points < 0
            for points in initial_constructor_points.values()
        ):
            raise ValueError("Constructor championship points must be finite and non-negative")
        if any(
            not isinstance(value, int) or value < 0
            for value in (current_driver_wins or {}).values()
        ):
            raise ValueError("Current driver win counts must be non-negative integers")
        if any(
            not isinstance(value, int) or value < 0
            for value in (current_constructor_wins or {}).values()
        ):
            raise ValueError("Current constructor win counts must be non-negative integers")
        constructors.update(initial_constructor_points)
        driver_totals: dict[str, list[float]] = {driver: [] for driver in drivers}
        driver_positions: dict[str, list[int]] = {driver: [] for driver in drivers}
        constructor_totals: dict[str, list[float]] = {
            team: [] for team in constructors
        }
        constructor_positions: dict[str, list[int]] = {
            team: [] for team in constructors
        }
        rng = np.random.default_rng(random_seed)

        for _ in range(num_simulations):
            points = {driver: float(current_driver_points.get(driver, 0.0)) for driver in drivers}
            win_counts = {
                driver: int((current_driver_wins or {}).get(driver, 0))
                for driver in drivers
            }
            team_points = {
                team: float(initial_constructor_points.get(team, 0.0))
                for team in constructors
            }
            team_wins = {
                team: int((current_constructor_wins or {}).get(team, 0))
                for team in constructors
            }
            for race in remaining_races:
                race_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                race_results = self.race_simulator.simulate(
                    race.strengths, race.dnf_probabilities, race_seed
                )
                race_score = self.scoring_engine.score_results(
                    race_results,
                    race.season,
                    constructor_by_driver=dict(constructor_by_driver or {}),
                    fastest_lap_driver_id=self._sample_fastest_lap(race, rng),
                )
                self._add_scores(points, team_points, race_score)
                winner_result = next(
                    (result for result in race_results if result["classified"]), None
                )
                winner = winner_result["driver_id"] if winner_result is not None else None
                if isinstance(winner, str):
                    win_counts[winner] = win_counts.get(winner, 0) + 1
                    winner_constructor = (constructor_by_driver or {}).get(winner)
                    if winner_constructor is not None:
                        team_wins[winner_constructor] = (
                            team_wins.get(winner_constructor, 0) + 1
                        )

                if race.sprint_strengths:
                    sprint_results = self.race_simulator.simulate(
                        race.sprint_strengths,
                        race.sprint_dnf_probabilities,
                        int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                    sprint_score = self.scoring_engine.score_results(
                        sprint_results,
                        race.season,
                        race_type="sprint",
                        constructor_by_driver=dict(constructor_by_driver or {}),
                    )
                    self._add_scores(points, team_points, sprint_score)

            ranking = sorted(
                drivers,
                key=lambda driver: (-points[driver], -win_counts[driver], driver),
            )
            positions = {
                driver: position for position, driver in enumerate(ranking, start=1)
            }
            for driver in drivers:
                driver_totals[driver].append(points[driver])
                driver_positions[driver].append(positions[driver])
            for team in constructors:
                constructor_totals[team].append(team_points[team])
            team_ranking = sorted(
                constructors,
                key=lambda team: (-team_points[team], -team_wins[team], team),
            )
            for position, team in enumerate(team_ranking, start=1):
                constructor_positions[team].append(position)

        driver_summary = self._summarize_championship(
            driver_totals, driver_positions
        )
        output: dict[str, Any] = {
            "simulations": num_simulations,
            "random_seed": random_seed,
            "driver_championship": driver_summary,
        }
        if constructors:
            output["constructor_championship"] = self._summarize_championship(
                constructor_totals, constructor_positions
            )
        return output

    @staticmethod
    def _add_scores(
        driver_points: dict[str, float],
        constructor_points: dict[str, float],
        score: dict[str, object],
    ) -> None:
        for driver, value in score["driver_points"].items():
            driver_points[driver] = driver_points.get(driver, 0.0) + float(value)
        for constructor, value in score["constructor_points"].items():
            constructor_points[constructor] = (
                constructor_points.get(constructor, 0.0) + float(value)
            )

    @staticmethod
    def _sample_fastest_lap(
        race: RaceForecast, rng: np.random.Generator
    ) -> str | None:
        if not race.fastest_lap_probabilities:
            return None
        drivers = list(race.fastest_lap_probabilities)
        probabilities = np.asarray(
            [race.fastest_lap_probabilities[driver] for driver in drivers],
            dtype=float,
        )
        return str(rng.choice(drivers, p=probabilities))

    @staticmethod
    def _summarize_championship(
        points: dict[str, list[float]], positions: dict[str, list[int]]
    ) -> dict[str, Any]:
        simulations = len(next(iter(points.values())))
        return {
            driver: {
                "winner_probability": sum(position == 1 for position in positions[driver])
                / simulations,
                "position_probabilities": {
                    str(position): positions[driver].count(position) / simulations
                    for position in range(1, len(positions) + 1)
                },
                "expected_points": float(np.mean(points[driver])),
                "median_points": float(np.median(points[driver])),
                "expected_final_position": float(np.mean(positions[driver])),
                "position_distribution": {
                    str(position): positions[driver].count(position)
                    for position in range(1, len(positions) + 1)
                },
            }
            for driver in points
        }
