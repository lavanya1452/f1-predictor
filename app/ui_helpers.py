"""Shared cached data and forecast helpers for the Streamlit pages."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loaders import load_json, load_jsonl
from src.simulation.championship import ChampionshipSimulator, RaceForecast
from src.simulation.race import RaceSimulator
from src.utils.config import AppConfig

TABLE_NAMES = {
    "seasons",
    "races",
    "circuits",
    "drivers",
    "constructors",
    "qualifying_results",
    "race_results",
    "driver_standings",
    "constructor_standings",
}


@st.cache_data(ttl=300, max_entries=16)
def load_processed_table(name: str) -> list[dict[str, Any]]:
    if name not in TABLE_NAMES:
        raise ValueError(f"Unknown processed table: {name}")
    return load_jsonl(AppConfig.from_env().processed_data_dir / f"{name}.jsonl")


@st.cache_data(ttl=300, max_entries=4)
def load_manifest() -> dict[str, Any] | None:
    path = AppConfig.from_env().processed_data_dir / "manifest.json"
    if not path.is_file():
        return None
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a manifest object in {path}")
    return value


@st.cache_data(ttl=300, max_entries=4)
def load_evaluation_report() -> dict[str, Any] | None:
    path = AppConfig.from_env().processed_data_dir / "baseline_evaluation.json"
    if not path.is_file():
        return None
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an evaluation report object in {path}")
    return value


@st.cache_data(ttl=600, max_entries=32)
def simulate_race_distribution(
    predicted_positions: tuple[tuple[str, float], ...],
    dnf_probabilities: tuple[tuple[str, float], ...],
    simulations: int,
    random_seed: int,
) -> pd.DataFrame:
    if simulations < 1:
        raise ValueError("Simulation count must be at least one")
    strengths = {
        driver: 1.0 / float(position)
        for driver, position in predicted_positions
    }
    if any(
        not np.isfinite(position) or not 1 <= position <= len(predicted_positions)
        for _, position in predicted_positions
    ):
        raise ValueError(
            "Expected finishing positions must be finite and within the entry count"
        )
    dnf = dict(dnf_probabilities)
    expected: defaultdict[str, list[int]] = defaultdict(list)
    wins: Counter[str] = Counter()
    dnfs: Counter[str] = Counter()
    rng = np.random.default_rng(random_seed)
    simulator = RaceSimulator()
    for _ in range(simulations):
        results = simulator.simulate(
            strengths,
            dnf,
            random_seed=int(rng.integers(0, 2**31 - 1)),
        )
        for result in results:
            driver = str(result["driver_id"])
            expected[driver].append(int(result["position"]))
            if result["position"] == 1 and result["classified"]:
                wins[driver] += 1
            if not result["classified"]:
                dnfs[driver] += 1
    return pd.DataFrame(
        [
            {
                "driver_id": driver,
                "expected_position": sum(positions) / simulations,
                "p1_probability": wins[driver] / simulations,
                "dnf_probability": dnfs[driver] / simulations,
                **{
                    f"p{position}_probability": sum(
                        value == position for value in expected[driver]
                    )
                    / simulations
                    for position in range(1, len(strengths) + 1)
                },
            }
            for driver, positions in expected.items()
        ]
    ).sort_values(["expected_position", "driver_id"])


def position_strengths(
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for prediction in predictions:
        driver = prediction.get("driver_id")
        expected_position = prediction.get("expected_position")
        if not isinstance(driver, str) or not isinstance(expected_position, (int, float)):
            raise ValueError("Evaluation report contains an invalid race prediction")
        if expected_position <= 0:
            raise ValueError(f"Invalid expected position for {driver}")
        strengths[driver] = 1.0 / expected_position
    if not strengths:
        raise ValueError("Evaluation fold has no race predictions")
    return strengths


def get_event_fold(
    report: dict[str, Any], model_key: str, season: int, round_number: int
) -> dict[str, Any] | None:
    model_report = report.get(model_key)
    if not isinstance(model_report, dict):
        return None
    folds = model_report.get("folds", [])
    return next(
        (
            fold
            for fold in folds
            if fold.get("season") == season and fold.get("round") == round_number
        ),
        None,
    )


def simulation_forecasts(
    future_races: list[dict[str, Any]],
    last_prediction_fold: dict[str, Any],
    dnf_fold: dict[str, Any] | None,
    performance_multiplier: dict[str, float] | None = None,
    dnf_override: dict[str, float] | None = None,
) -> list[RaceForecast]:
    predictions = last_prediction_fold.get("predictions", [])
    strengths = position_strengths(predictions)
    multipliers = performance_multiplier or {}
    for driver, multiplier in multipliers.items():
        if driver in strengths:
            if not 0.1 <= multiplier <= 3.0:
                raise ValueError("Performance multipliers must be between 0.1 and 3.0")
            strengths[driver] *= multiplier
    dnf_values: dict[str, float] = {}
    if dnf_fold:
        for row in dnf_fold.get("predictions", []):
            probability = row.get("calibrated_dnf_probability")
            driver = row.get("driver_id")
            if isinstance(driver, str) and isinstance(probability, (int, float)):
                dnf_values[driver] = float(probability)
    if dnf_override:
        for driver, probability in dnf_override.items():
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid DNF probability for {driver}")
            if driver in strengths:
                dnf_values[driver] = probability

    return [
        RaceForecast(
            season=int(race["season"]),
            round=int(race["round"]),
            strengths=strengths,
            dnf_probabilities=dnf_values,
        )
        for race in future_races
    ]


def current_season_standings(
    season: int,
    through_round: int,
) -> tuple[dict[str, float], dict[str, float]]:
    driver_rows = [
        row
        for row in load_processed_table("driver_standings")
        if row["season"] == season and row["round"] == through_round
    ]
    constructor_rows = [
        row
        for row in load_processed_table("constructor_standings")
        if row["season"] == season and row["round"] == through_round
    ]
    if not driver_rows:
        raise ValueError(
            f"No driver standings snapshot exists after round {through_round} in {season}"
        )
    return (
        {row["driver_id"]: float(row["points"]) for row in driver_rows},
        {row["constructor_id"]: float(row["points"]) for row in constructor_rows},
    )


def current_constructor_map(
    season: int,
    through_round: int,
) -> dict[str, str]:
    results = [
        row
        for row in load_processed_table("race_results")
        if row["season"] < season
        or (row["season"] == season and row["round"] <= through_round)
    ]
    mapping: dict[str, str] = {}
    for row in sorted(results, key=lambda record: (record["season"], record["round"])):
        mapping[row["driver_id"]] = row["constructor_id"]
    return mapping


@st.cache_data(ttl=600, max_entries=32)
def run_championship_simulation(
    current_driver_points: dict[str, float],
    current_constructor_points: dict[str, float],
    constructor_by_driver: dict[str, str],
    forecasts: list[dict[str, Any]],
    simulation_count: int,
    random_seed: int,
) -> dict[str, Any]:
    race_forecasts = [
        RaceForecast(
            season=int(forecast["season"]),
            round=int(forecast["round"]),
            strengths=forecast["strengths"],
            dnf_probabilities=forecast.get("dnf_probabilities", {}),
        )
        for forecast in forecasts
    ]
    return ChampionshipSimulator().simulate(
        current_driver_points=current_driver_points,
        current_constructor_points=current_constructor_points,
        constructor_by_driver=constructor_by_driver,
        remaining_races=race_forecasts,
        num_simulations=simulation_count,
        random_seed=random_seed,
    )


def forecast_dicts(forecasts: list[RaceForecast]) -> list[dict[str, Any]]:
    return [
        {
            "season": forecast.season,
            "round": forecast.round,
            "strengths": dict(forecast.strengths),
            "dnf_probabilities": dict(forecast.dnf_probabilities),
        }
        for forecast in forecasts
    ]


def prepare_simulation_context(
    season: int, through_round: int, report: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    prediction_fold = get_event_fold(report, "xgboost_position", season, through_round)
    if prediction_fold is None:
        raise ValueError("No walk-forward prediction exists for the selected cutoff race")
    races = load_processed_table("races")
    future_races = [
        race
        for race in races
        if race["season"] == season and race["round"] > through_round
    ]
    if not future_races:
        raise ValueError("There are no remaining scheduled races after this cutoff")
    driver_points, constructor_points = current_season_standings(season, through_round)
    constructor_map = current_constructor_map(season, through_round)
    prediction_drivers = {
        row["driver_id"] for row in prediction_fold.get("predictions", [])
    }
    missing = sorted(set(driver_points) - prediction_drivers)
    if missing:
        raise ValueError(
            "Cutoff-race model predictions do not cover all drivers in the standings: "
            + ", ".join(missing)
        )
    dnf_fold = get_event_fold(
        report, "xgboost_dnf_probability", season, through_round
    )
    forecasts = simulation_forecasts(future_races, prediction_fold, dnf_fold)
    no_dnf_calibration = not any(
        "calibrated_dnf_probability" in row
        for row in (dnf_fold or {}).get("predictions", [])
    )
    return (
        {
            "driver_points": driver_points,
            "constructor_points": constructor_points,
            "constructor_by_driver": constructor_map,
            "forecasts": forecasts,
        },
        (
            "No calibrated DNF model is available at this cutoff; the simulation "
            "assumes zero DNF probability."
            if no_dnf_calibration
            else None
        ),
    )
