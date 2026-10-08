import pandas as pd

from src.features.engineering import FEATURE_COLUMNS
from src.models.evaluation import evaluate_baselines, regression_metrics


def _frame() -> pd.DataFrame:
    rows = []
    for round_number in range(1, 4):
        row = {feature: 1.0 for feature in FEATURE_COLUMNS}
        row.update(
            {
                "season": 2024,
                "round": round_number,
                "driver_id": "driver-a",
                "constructor_id": "team-a",
                "circuit_id": "circuit-a",
                "driver_previous_finish": float(round_number),
                "driver_rolling_finish_mean_5": float(round_number),
                "target_position": float(4 - round_number),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_walk_forward_uses_only_prior_race_events() -> None:
    results = evaluate_baselines(_frame(), minimum_training_events=1)

    assert results["validation"] == "expanding walk-forward by race event"
    assert results["fold_count"] == 2
    assert [fold["round"] for fold in results["folds"]] == [2, 3]
    assert set(results["overall"]) == {
        "previous_result",
        "rolling_average",
        "sklearn_random_forest",
    }


def test_regression_metrics_are_correct() -> None:
    assert regression_metrics([1.0, 3.0], [2.0, 1.0]) == {
        "mae": 1.5,
        "rmse": 1.5811388300841898,
    }
