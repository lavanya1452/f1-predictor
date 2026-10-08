"""Reproducible XGBoost training with chronological DNF probability calibration."""

import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.features.engineering import FEATURE_COLUMNS
from src.models.evaluation import _preprocessor, regression_metrics
from src.utils.config import AppConfig


class ProbabilityEstimator(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


def _xgb_options(config: AppConfig) -> dict[str, Any]:
    parameters = config.model_parameters
    return {
        "n_estimators": int(parameters.get("xgb_n_estimators", 300)),
        "max_depth": int(parameters.get("xgb_max_depth", 4)),
        "learning_rate": float(parameters.get("xgb_learning_rate", 0.05)),
        "subsample": float(parameters.get("xgb_subsample", 0.8)),
        "colsample_bytree": float(parameters.get("xgb_colsample_bytree", 0.8)),
        "reg_lambda": float(parameters.get("xgb_reg_lambda", 1.0)),
        "random_state": config.random_seed,
        "n_jobs": int(parameters.get("xgb_n_jobs", 1)),
    }


def make_position_estimator(config: AppConfig) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    **_xgb_options(config),
                ),
            ),
        ]
    )


def make_dnf_estimator(config: AppConfig) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    **_xgb_options(config),
                ),
            ),
        ]
    )


@dataclass
class CalibratedDNFModel:
    """Final XGBoost DNF estimator followed by a fitted sigmoid calibrator."""

    estimator: ProbabilityEstimator
    calibrator: LogisticRegression

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(features)[:, 1]
        calibrated = self.calibrator.predict_proba(logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1))[
            :, 1
        ]
        return np.column_stack((1.0 - calibrated, calibrated))


def fit_calibrated_dnf(
    frame: pd.DataFrame, config: AppConfig | None = None
) -> CalibratedDNFModel | None:
    """Fit a DNF classifier and sigmoid calibrator using chronological OOF scores."""
    settings = config or AppConfig.from_env()
    training = frame.loc[frame["target_dnf"].notna()].copy()
    if training["target_dnf"].nunique() != 2:
        return None
    events = sorted(set(zip(training["season"].astype(int), training["round"].astype(int))))
    if len(events) < 4:
        return None
    splitter = TimeSeriesSplit(n_splits=3)
    event_array = np.asarray(events, dtype=object)
    out_of_fold_probability: list[float] = []
    out_of_fold_target: list[int] = []
    for train_indices, validation_indices in splitter.split(event_array):
        train_events = [events[index] for index in train_indices]
        validation_events = [events[index] for index in validation_indices]
        train_mask = [
            (int(season), int(round_number)) in train_events
            for season, round_number in zip(training["season"], training["round"])
        ]
        validation_mask = [
            (int(season), int(round_number)) in validation_events
            for season, round_number in zip(training["season"], training["round"])
        ]
        train = training.loc[train_mask]
        validation = training.loc[validation_mask]
        if train["target_dnf"].nunique() != 2 or validation.empty:
            continue
        estimator = make_dnf_estimator(settings)
        estimator.fit(train[FEATURE_COLUMNS], train["target_dnf"].astype(int))
        out_of_fold_probability.extend(
            estimator.predict_proba(validation[FEATURE_COLUMNS])[:, 1].tolist()
        )
        out_of_fold_target.extend(validation["target_dnf"].astype(int).tolist())
    if len(set(out_of_fold_target)) != 2:
        return None
    calibration = LogisticRegression(random_state=settings.random_seed)
    scores = logit(np.clip(out_of_fold_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    calibration.fit(scores, out_of_fold_target)
    final_estimator = make_dnf_estimator(settings)
    final_estimator.fit(training[FEATURE_COLUMNS], training["target_dnf"].astype(int))
    return CalibratedDNFModel(final_estimator, calibration)


def train_models(
    frame: pd.DataFrame, config: AppConfig | None = None
) -> dict[str, Any]:
    """Train available models on real processed rows and save reproducible artifacts."""
    settings = config or AppConfig.from_env()
    classified = frame.loc[frame["target_position"].notna()].copy()
    if classified.empty:
        raise ValueError("Cannot train a position model without classified race results")
    position_model = make_position_estimator(settings)
    position_model.fit(
        classified[FEATURE_COLUMNS], classified["target_position"].astype(float)
    )
    points_rows = frame.loc[frame["target_points"].notna()].copy()
    if points_rows.empty:
        raise ValueError("Cannot train a points model without race points records")
    points_model = make_position_estimator(settings)
    points_model.fit(
        points_rows[FEATURE_COLUMNS], points_rows["target_points"].astype(float)
    )
    dnf_model = fit_calibrated_dnf(frame, settings)
    last_event = max(
        zip(classified["season"].astype(int), classified["round"].astype(int))
    )
    artifact_dir = settings.model_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    position_path = artifact_dir / "position_xgboost.joblib"
    joblib.dump(position_model, position_path)
    points_path = artifact_dir / "points_xgboost.joblib"
    joblib.dump(points_model, points_path)
    artifacts: dict[str, Any] = {
        "position_model": str(position_path),
        "points_model": str(points_path),
        "dnf_model": None,
    }
    if dnf_model is not None:
        dnf_path = artifact_dir / "dnf_xgboost_sigmoid.joblib"
        joblib.dump(dnf_model, dnf_path)
        artifacts["dnf_model"] = str(dnf_path)
    metadata = {
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_rows": len(classified),
        "points_training_rows": len(points_rows),
        "dnf_training_rows": int(frame["target_dnf"].notna().sum()),
        "last_training_event": {"season": last_event[0], "round": last_event[1]},
        "features": FEATURE_COLUMNS,
        "random_seed": settings.random_seed,
        "xgboost_parameters": _xgb_options(settings),
        "dnf_probability_calibration": (
            "sigmoid using expanding time-series out-of-fold scores"
            if dnf_model is not None
            else "not trained: insufficient chronological data or both classes"
        ),
    }
    metadata_path = artifact_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifacts["metadata"] = str(metadata_path)
    return artifacts


def evaluate_xgboost_walk_forward(
    frame: pd.DataFrame,
    minimum_training_events: int = 5,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Report race-by-race expanding-window position-model metrics."""
    return _evaluate_xgboost_regression_target(
        frame,
        target_column="target_position",
        prediction_column="expected_position",
        minimum_training_events=minimum_training_events,
        config=config,
    )


def evaluate_points_walk_forward(
    frame: pd.DataFrame,
    minimum_training_events: int = 5,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Report race-by-race expanding-window driver-points metrics."""
    return _evaluate_xgboost_regression_target(
        frame,
        target_column="target_points",
        prediction_column="expected_points",
        minimum_training_events=minimum_training_events,
        config=config,
    )


def _evaluate_xgboost_regression_target(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    minimum_training_events: int,
    config: AppConfig | None,
) -> dict[str, Any]:
    if minimum_training_events < 1:
        raise ValueError("minimum_training_events must be at least one")
    settings = config or AppConfig.from_env()
    training_data = frame.loc[frame[target_column].notna()].copy()
    if training_data.empty:
        raise ValueError(f"No training targets are available in {target_column}")
    events = sorted(
        set(zip(training_data["season"].astype(int), training_data["round"].astype(int)))
    )
    actual_values: list[float] = []
    predicted_values: list[float] = []
    folds: list[dict[str, Any]] = []
    for event in events:
        train = training_data.loc[
            (training_data["season"] < event[0])
            | ((training_data["season"] == event[0]) & (training_data["round"] < event[1]))
        ]
        train_events = set(zip(train["season"].astype(int), train["round"].astype(int)))
        if len(train_events) < minimum_training_events:
            continue
        test = training_data.loc[
            (training_data["season"] == event[0]) & (training_data["round"] == event[1])
        ]
        estimator = make_position_estimator(settings)
        estimator.fit(train[FEATURE_COLUMNS], train[target_column].astype(float))
        predicted_array = estimator.predict(test[FEATURE_COLUMNS]).astype(float)
        if target_column == "target_position":
            predicted_array = np.clip(predicted_array, 1.0, float(len(test)))
        elif target_column == "target_points":
            predicted_array = np.maximum(predicted_array, 0.0)
        predicted = predicted_array.tolist()
        actual = test[target_column].astype(float).tolist()
        folds.append(
            {
                "season": event[0],
                "round": event[1],
                "samples": len(test),
                "metrics": regression_metrics(actual, predicted),
                "predictions": [
                    {
                        "driver_id": row["driver_id"],
                        "constructor_id": row["constructor_id"],
                        "circuit_id": row["circuit_id"],
                        prediction_column: float(prediction),
                        f"actual_{prediction_column.removeprefix('expected_')}": float(target),
                    }
                    for (_, row), prediction, target in zip(
                        test.iterrows(), predicted, actual
                    )
                ],
            }
        )
        actual_values.extend(actual)
        predicted_values.extend(predicted)
    if not folds:
        raise ValueError("No XGBoost walk-forward folds met the requested history requirement")
    return {
        "validation": "expanding walk-forward by race event",
        "target": target_column,
        "minimum_training_events": minimum_training_events,
        "fold_count": len(folds),
        "overall": regression_metrics(actual_values, predicted_values),
        "folds": folds,
    }


def evaluate_dnf_walk_forward(
    frame: pd.DataFrame,
    minimum_training_events: int = 5,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Compare raw and sigmoid-calibrated DNF probabilities out of sample."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    settings = config or AppConfig.from_env()
    events = sorted(set(zip(frame["season"].astype(int), frame["round"].astype(int))))
    actual_all: list[int] = []
    raw_all: list[float] = []
    calibrated_actual: list[int] = []
    calibrated_all: list[float] = []
    calibrated_events = 0
    folds: list[dict[str, Any]] = []
    for event in events:
        train = frame.loc[
            (frame["season"] < event[0])
            | ((frame["season"] == event[0]) & (frame["round"] < event[1]))
        ]
        train = train.loc[train["target_dnf"].notna()]
        train_events = set(zip(train["season"].astype(int), train["round"].astype(int)))
        if len(train_events) < minimum_training_events or train["target_dnf"].nunique() != 2:
            continue
        test = frame.loc[
            (frame["season"] == event[0]) & (frame["round"] == event[1])
        ]
        test = test.loc[test["target_dnf"].notna()]
        if test.empty:
            continue
        calibrated_model = fit_calibrated_dnf(train, settings)
        raw_estimator = (
            calibrated_model.estimator if calibrated_model is not None else make_dnf_estimator(settings)
        )
        if calibrated_model is None:
            raw_estimator.fit(train[FEATURE_COLUMNS], train["target_dnf"].astype(int))
        raw_probability = raw_estimator.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        actual = test["target_dnf"].astype(int).tolist()
        fold: dict[str, Any] = {
            "season": event[0],
            "round": event[1],
            "samples": len(test),
            "raw_brier_score": float(brier_score_loss(actual, raw_probability)),
            "raw_log_loss": float(log_loss(actual, raw_probability, labels=[0, 1])),
        }
        actual_all.extend(actual)
        raw_all.extend(raw_probability.tolist())
        if calibrated_model is not None:
            calibrated_probability = calibrated_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            calibrated_actual.extend(actual)
            calibrated_all.extend(calibrated_probability.tolist())
            fold["calibrated_brier_score"] = float(
                brier_score_loss(actual, calibrated_probability)
            )
            fold["calibrated_log_loss"] = float(
                log_loss(actual, calibrated_probability, labels=[0, 1])
            )
            calibrated_events += 1
            fold["predictions"] = [
                {
                    "driver_id": row["driver_id"],
                    "raw_dnf_probability": float(raw_probability[index]),
                    "calibrated_dnf_probability": float(calibrated_probability[index]),
                    "actual_dnf": actual[index],
                }
                for index, (_, row) in enumerate(test.iterrows())
            ]
        else:
            fold["predictions"] = [
                {
                    "driver_id": row["driver_id"],
                    "raw_dnf_probability": float(raw_probability[index]),
                    "actual_dnf": actual[index],
                }
                for index, (_, row) in enumerate(test.iterrows())
            ]
        folds.append(fold)
    if not folds:
        raise ValueError("No DNF walk-forward folds met the requested history requirement")
    raw_metrics: dict[str, float | None] = {
        "brier_score": float(brier_score_loss(actual_all, raw_all)),
        "log_loss": float(log_loss(actual_all, raw_all, labels=[0, 1])),
        "roc_auc": (
            float(roc_auc_score(actual_all, raw_all))
            if len(set(actual_all)) == 2
            else None
        ),
    }
    output: dict[str, Any] = {
        "validation": "expanding walk-forward by race event",
        "fold_count": len(folds),
        "raw_probability_metrics": raw_metrics,
        "calibration_folds": calibrated_events,
        "folds": folds,
    }
    if calibrated_all:
        output["calibrated_probability_metrics"] = {
            "brier_score": float(brier_score_loss(calibrated_actual, calibrated_all)),
            "log_loss": float(log_loss(calibrated_actual, calibrated_all, labels=[0, 1])),
            "roc_auc": (
                float(roc_auc_score(calibrated_actual, calibrated_all))
                if len(set(calibrated_actual)) == 2
                else None
            ),
        }
    return output
