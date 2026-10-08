"""Chronological walk-forward evaluation and baselines."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.engineering import FEATURE_COLUMNS
from src.utils.config import AppConfig


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must be non-empty and equal length")
    errors = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _preprocessor() -> ColumnTransformer:
    categorical = ["driver_id", "constructor_id", "circuit_id"]
    numeric = [name for name in FEATURE_COLUMNS if name not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _sklearn_baseline(config: AppConfig) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=2,
                    random_state=config.random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_baselines(
    frame: pd.DataFrame,
    minimum_training_events: int = 1,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Evaluate simple baselines with expanding, race-ordered training windows."""
    if minimum_training_events < 1:
        raise ValueError("minimum_training_events must be at least one")
    required = set(FEATURE_COLUMNS + ["target_position"])
    if not required.issubset(frame.columns):
        raise ValueError(f"Feature frame is missing columns: {sorted(required - set(frame.columns))}")
    eligible = frame.loc[frame["target_position"].notna()].copy()
    if eligible.empty:
        raise ValueError("No classified finishing positions are available for evaluation")
    events = sorted(set(zip(eligible["season"].astype(int), eligible["round"].astype(int))))
    settings = config or AppConfig.from_env()
    fold_results: list[dict[str, Any]] = []
    predictions_by_model: dict[str, dict[str, list[float]]] = {
        "previous_result": {"actual": [], "predicted": []},
        "rolling_average": {"actual": [], "predicted": []},
        "sklearn_random_forest": {"actual": [], "predicted": []},
    }
    for event in events:
        prior = eligible.loc[
            (eligible["season"] < event[0])
            | ((eligible["season"] == event[0]) & (eligible["round"] < event[1]))
        ]
        prior_event_count = len(
            set(zip(prior["season"].astype(int), prior["round"].astype(int)))
        )
        if prior_event_count < minimum_training_events:
            continue
        test = eligible.loc[
            (eligible["season"] == event[0]) & (eligible["round"] == event[1])
        ]
        actual = test["target_position"].astype(float).tolist()
        fallback = float(prior["target_position"].median())
        fold_predictions: dict[str, list[float]] = {}
        for name, feature in (
            ("previous_result", "driver_previous_finish"),
            ("rolling_average", "driver_rolling_finish_mean_5"),
        ):
            prediction = test[feature].fillna(fallback).astype(float).tolist()
            fold_predictions[name] = prediction

        estimator = _sklearn_baseline(settings)
        estimator.fit(prior[FEATURE_COLUMNS], prior["target_position"].astype(float))
        fold_predictions["sklearn_random_forest"] = estimator.predict(
            test[FEATURE_COLUMNS]
        ).astype(float).tolist()
        fold: dict[str, Any] = {
            "season": event[0],
            "round": event[1],
            "samples": len(test),
            "metrics": {},
        }
        for name, prediction in fold_predictions.items():
            fold["metrics"][name] = regression_metrics(actual, prediction)
            predictions_by_model[name]["actual"].extend(actual)
            predictions_by_model[name]["predicted"].extend(prediction)
        fold_results.append(fold)

    if not fold_results:
        raise ValueError(
            "Not enough chronological race events for the requested training window"
        )
    overall = {
        name: regression_metrics(values["actual"], values["predicted"])
        for name, values in predictions_by_model.items()
    }
    return {
        "validation": "expanding walk-forward by race event",
        "minimum_training_events": minimum_training_events,
        "fold_count": len(fold_results),
        "overall": overall,
        "folds": fold_results,
    }
