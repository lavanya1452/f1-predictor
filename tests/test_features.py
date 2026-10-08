from copy import deepcopy
from typing import Any

from src.features.engineering import FEATURE_COLUMNS, FeatureBuilder


def _history() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    races = [
        {
            "season": 2024,
            "round": round_number,
            "race_id": f"2024-{round_number}",
            "circuit_id": "circuit-a" if round_number != 3 else "circuit-b",
            "name": f"Grand Prix {round_number}",
            "date": f"2024-03-0{round_number}",
        }
        for round_number in range(1, 4)
    ]
    qualifying = [
        {
            "season": 2024,
            "round": round_number,
            "race_id": f"2024-{round_number}",
            "driver_id": "driver-a",
            "constructor_id": "team-a",
            "position": round_number,
        }
        for round_number in range(1, 4)
    ]
    race_results = [
        {
            "season": 2024,
            "round": round_number,
            "race_id": f"2024-{round_number}",
            "driver_id": "driver-a",
            "constructor_id": "team-a",
            "grid": round_number,
            "position": round_number,
            "position_text": str(round_number),
            "points": 26 - round_number,
            "laps": 57,
            "status": "Finished",
        }
        for round_number in range(1, 4)
    ]
    driver_standings = [
        {
            "season": 2024,
            "round": round_number,
            "driver_id": "driver-a",
            "position": 1,
            "points": 26 * round_number,
            "wins": round_number,
        }
        for round_number in range(1, 4)
    ]
    constructor_standings = [
        {
            "season": 2024,
            "round": round_number,
            "constructor_id": "team-a",
            "position": 1,
            "points": 26 * round_number,
            "wins": round_number,
        }
        for round_number in range(1, 4)
    ]
    return races, qualifying, race_results, driver_standings, constructor_standings


def _build(history: tuple[list[dict[str, Any]], ...]):
    return FeatureBuilder.build_training_frame(
        races=history[0],
        qualifying_results=history[1],
        race_results=history[2],
        driver_standings=history[3],
        constructor_standings=history[4],
    )


def test_features_use_only_information_available_before_each_race() -> None:
    base = _history()
    altered = deepcopy(base)
    altered[2][1]["position"] = 1
    altered[2][1]["points"] = 99
    altered[2][1]["status"] = "Engine"
    altered[2][1]["laps"] = 0
    altered[2][2]["position"] = 1
    altered[2][2]["points"] = 100
    altered[2][2]["status"] = "Collision"
    altered[3][1]["points"] = 500
    altered[4][1]["points"] = 500

    before = _build(base).query("round == 2").iloc[0]
    after = _build(altered).query("round == 2").iloc[0]

    for feature in FEATURE_COLUMNS:
        assert before[feature] == after[feature] or (
            before[feature] != before[feature] and after[feature] != after[feature]
        ), feature
    assert before["driver_points_entering_race"] == 26
    assert before["constructor_points_entering_race"] == 26
    assert before["target_position"] == 2
    assert after["target_position"] == 1


def test_rolling_circuit_and_qualifying_features_exclude_current_result() -> None:
    frame = _build(_history())
    second_race = frame.query("round == 2").iloc[0]
    third_race = frame.query("round == 3").iloc[0]

    assert second_race["driver_previous_finish"] == 1
    assert second_race["driver_rolling_finish_mean_5"] == 1
    assert second_race["driver_previous_qualifying_position"] == 1
    assert second_race["qualifying_position"] == 2
    assert second_race["driver_circuit_starts"] == 1
    assert third_race["driver_circuit_starts"] == 0
    assert third_race["driver_points_entering_race"] == 52
