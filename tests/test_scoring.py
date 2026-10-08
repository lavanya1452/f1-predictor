from pathlib import Path

import pytest

from src.scoring.engine import ScoringConfig, ScoringEngine, SeasonScoringRules


def test_2024_race_scoring_and_fastest_lap_points() -> None:
    results = [
        {
            "driver_id": "driver-a",
            "constructor_id": "team-a",
            "position": 1,
            "status": "Finished",
            "classified": True,
        },
        {
            "driver_id": "driver-b",
            "constructor_id": "team-a",
            "position": 2,
            "status": "Finished",
            "classified": True,
        },
        {
            "driver_id": "driver-c",
            "constructor_id": "team-b",
            "position": 3,
            "status": "DNF",
            "classified": False,
        },
    ]

    score = ScoringEngine().score_results(
        results, 2024, fastest_lap_driver_id="driver-b"
    )

    assert score["driver_points"] == {
        "driver-a": 25.0,
        "driver-b": 19.0,
        "driver-c": 0.0,
    }
    assert score["constructor_points"] == {"team-a": 44.0, "team-b": 0.0}


def test_season_specific_rules_and_sprint_points() -> None:
    engine = ScoringEngine()
    classic_result = [{"driver_id": "driver-a", "position": 1, "classified": True}]
    sprint_result = [{"driver_id": "driver-a", "position": 1, "classified": True}]

    assert engine.score_results(classic_result, 2008)["driver_points"]["driver-a"] == 10.0
    assert (
        engine.score_results(sprint_result, 2024, race_type="sprint")["driver_points"][
            "driver-a"
        ]
        == 8.0
    )


def test_fastest_lap_rules_follow_season() -> None:
    engine = ScoringEngine()
    finishers = [
        {
            "driver_id": f"driver-{position}",
            "position": position,
            "classified": True,
        }
        for position in range(1, 12)
    ]
    early_score = engine.score_results(
        finishers, 1958, fastest_lap_driver_id="driver-11"
    )
    no_fastest_lap_score = engine.score_results(
        finishers, 1960, fastest_lap_driver_id="driver-11"
    )
    restricted_score = engine.score_results(
        finishers, 2024, fastest_lap_driver_id="driver-11"
    )

    assert early_score["driver_points"]["driver-11"] == 1.0
    assert no_fastest_lap_score["driver_points"]["driver-11"] == 0.0
    assert restricted_score["driver_points"]["driver-11"] == 0.0


def test_scoring_configuration_can_override_season_rules() -> None:
    config = ScoringConfig(
        seasons={
            2030: SeasonScoringRules(
                race_points=[12, 7],
                sprint_points=[3],
                fastest_lap_points=0,
            )
        }
    )

    score = ScoringEngine(config).score_results(
        [{"driver_id": "driver-a", "position": 1, "classified": True}], 2030
    )
    assert score["driver_points"]["driver-a"] == 12.0


def test_scoring_configuration_loads_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "scoring.json"
    config_path.write_text(
        '{"seasons":{"2030":{"race_points":[5],"sprint_points":[]}}}',
        encoding="utf-8",
    )
    score = ScoringEngine.from_file(config_path).score_results(
        [{"driver_id": "driver-a", "position": 1, "classified": True}],
        2030,
    )

    assert score["driver_points"]["driver-a"] == 5.0


def test_invalid_score_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="No scoring rules"):
        ScoringEngine().score_results([], 2040)
    with pytest.raises(ValueError, match="Duplicate result"):
        ScoringEngine().score_results(
            [
                {"driver_id": "driver-a", "position": 1, "classified": True},
                {"driver_id": "driver-a", "position": 2, "classified": True},
            ],
            2024,
        )
