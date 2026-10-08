import pytest

from src.simulation.championship import ChampionshipSimulator, RaceForecast
from src.simulation.race import RaceSimulator, plackett_luce_probabilities


def test_race_order_has_each_driver_once_and_valid_positions() -> None:
    strengths = {"driver-a": 4.0, "driver-b": 2.0, "driver-c": 1.0}
    results = RaceSimulator().simulate(strengths, random_seed=17)

    assert {result["driver_id"] for result in results} == set(strengths)
    assert len({result["driver_id"] for result in results}) == len(strengths)
    assert [result["position"] for result in results] == [1, 2, 3]
    assert all(result["classified_position"] is not None for result in results)


def test_race_simulation_is_reproducible_with_seed() -> None:
    simulator = RaceSimulator()
    strengths = {"driver-a": 4.0, "driver-b": 2.0, "driver-c": 1.0}

    assert simulator.simulate(strengths, random_seed=42) == simulator.simulate(
        strengths, random_seed=42
    )


def test_dnf_drivers_are_present_but_not_classified() -> None:
    results = RaceSimulator().simulate(
        {"driver-a": 4.0, "driver-b": 2.0, "driver-c": 1.0},
        {"driver-b": 1.0},
        random_seed=1,
    )
    dnf = next(result for result in results if result["driver_id"] == "driver-b")

    assert dnf["status"] == "DNF"
    assert dnf["classified"] is False
    assert dnf["classified_position"] is None
    assert len(results) == 3


def test_plackett_luce_selection_probabilities_sum_to_one() -> None:
    probabilities = plackett_luce_probabilities(
        {"driver-a": 4.0, "driver-b": 2.0, "driver-c": 1.0}
    )

    assert probabilities["driver-a"] == pytest.approx(4 / 7)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_invalid_simulation_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        RaceSimulator().simulate({"driver-a": 0.0})
    with pytest.raises(ValueError, match="Invalid DNF"):
        RaceSimulator().simulate({"driver-a": 1.0}, {"driver-a": 1.1})


def test_championship_monte_carlo_is_seeded_and_probabilities_are_normalized() -> None:
    simulator = ChampionshipSimulator()
    races = [
        RaceForecast(
            season=2024,
            round=2,
            strengths={"driver-a": 9.0, "driver-b": 1.0},
            dnf_probabilities={"driver-b": 0.2},
        )
    ]
    first = simulator.simulate(
        {"driver-a": 0.0, "driver-b": 0.0},
        races,
        num_simulations=100,
        random_seed=99,
        constructor_by_driver={"driver-a": "team-a", "driver-b": "team-b"},
    )
    second = simulator.simulate(
        {"driver-a": 0.0, "driver-b": 0.0},
        races,
        num_simulations=100,
        random_seed=99,
        constructor_by_driver={"driver-a": "team-a", "driver-b": "team-b"},
    )

    assert first == second
    driver_results = first["driver_championship"]
    assert sum(row["winner_probability"] for row in driver_results.values()) == pytest.approx(1)
    for result in driver_results.values():
        assert sum(result["position_probabilities"].values()) == pytest.approx(1)
    assert sum(
        row["winner_probability"]
        for row in first["constructor_championship"].values()
    ) == pytest.approx(1)


def test_championship_rejects_invalid_forecast_probabilities() -> None:
    race = RaceForecast(
        season=2024,
        round=1,
        strengths={"driver-a": 1.0},
        fastest_lap_probabilities={"driver-a": 0.5},
    )

    with pytest.raises(ValueError, match="sum to one"):
        ChampionshipSimulator().simulate(
            {"driver-a": 0.0}, [race], num_simulations=2
        )
