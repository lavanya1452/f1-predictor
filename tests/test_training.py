from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.engineering import FEATURE_COLUMNS
from src.models.training import (
    CalibratedDNFModel,
    evaluate_dnf_walk_forward,
    evaluate_points_walk_forward,
    evaluate_xgboost_walk_forward,
    fit_calibrated_dnf,
    train_models,
)
from src.utils.config import AppConfig


class FixedProbabilityEstimator:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(features), 0.7)
        return np.column_stack((1.0 - probabilities, probabilities))


def _dnf_training_frame() -> pd.DataFrame:
    rows = []
    for round_number in range(1, 9):
        for driver_number in range(2):
            row = {feature: 1.0 for feature in FEATURE_COLUMNS}
            row.update(
                {
                    "season": 2024,
                    "round": round_number,
                    "driver_id": f"driver-{driver_number}",
                    "constructor_id": f"team-{driver_number}",
                    "circuit_id": "circuit-a",
                    "target_dnf": (round_number + driver_number) % 2,
                    "target_position": float(driver_number + 1),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_sigmoid_dnf_predictions_are_normalized() -> None:
    calibration = LogisticRegression().fit(
        np.array([[-2.0], [-1.0], [1.0], [2.0]]), [0, 0, 1, 1]
    )
    model = CalibratedDNFModel(FixedProbabilityEstimator(), calibration)
    probabilities = model.predict_proba(pd.DataFrame(index=range(3)))

    assert probabilities.shape == (3, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_dnf_classifier_calibrates_with_chronological_out_of_fold_data() -> None:
    config = AppConfig(
        model_parameters={
            "xgb_n_estimators": 5,
            "xgb_max_depth": 2,
            "xgb_n_jobs": 1,
        }
    )
    calibrated = fit_calibrated_dnf(_dnf_training_frame(), config)

    assert calibrated is not None
    sample = _dnf_training_frame().iloc[:3][FEATURE_COLUMNS]
    assert np.allclose(calibrated.predict_proba(sample).sum(axis=1), 1.0)


def test_xgboost_models_train_and_regression_targets_evaluate_walk_forward(
    tmp_path: Path,
) -> None:
    frame = _dnf_training_frame()
    frame["target_points"] = frame["round"].astype(float) * 2
    config = AppConfig(
        project_root=tmp_path,
        model_parameters={
            "xgb_n_estimators": 5,
            "xgb_max_depth": 2,
            "xgb_n_jobs": 1,
        },
    )

    artifacts = train_models(frame, config)
    position_evaluation = evaluate_xgboost_walk_forward(
        frame, minimum_training_events=2, config=config
    )
    points_evaluation = evaluate_points_walk_forward(
        frame, minimum_training_events=2, config=config
    )
    dnf_evaluation = evaluate_dnf_walk_forward(
        frame, minimum_training_events=2, config=config
    )

    assert (config.model_dir / "position_xgboost.joblib").is_file()
    assert (config.model_dir / "points_xgboost.joblib").is_file()
    assert artifacts["dnf_model"] is not None
    assert position_evaluation["fold_count"] == 6
    assert points_evaluation["fold_count"] == 6
    assert "expected_points" in points_evaluation["folds"][0]["predictions"][0]
    assert "raw_probability_metrics" in dnf_evaluation
    assert "calibrated_probability_metrics" in dnf_evaluation
