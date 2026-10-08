"""Run baseline walk-forward evaluation on locally processed data."""

import argparse
import json

from src.features.engineering import FeatureBuilder
from src.models.evaluation import evaluate_baselines
from src.models.training import (
    evaluate_dnf_walk_forward,
    evaluate_points_walk_forward,
    evaluate_xgboost_walk_forward,
)
from src.utils.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate F1 baselines chronologically.")
    parser.add_argument("--minimum-training-events", type=int, default=1)
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Also evaluate XGBoost position and DNF probability models.",
    )
    args = parser.parse_args()
    config = AppConfig.from_env()
    frame = FeatureBuilder(config).build_from_processed()
    results = evaluate_baselines(
        frame,
        minimum_training_events=args.minimum_training_events,
        config=config,
    )
    if args.include_xgboost:
        results["xgboost_position"] = evaluate_xgboost_walk_forward(
            frame,
            minimum_training_events=args.minimum_training_events,
            config=config,
        )
        results["xgboost_points"] = evaluate_points_walk_forward(
            frame,
            minimum_training_events=args.minimum_training_events,
            config=config,
        )
        results["xgboost_dnf_probability"] = evaluate_dnf_walk_forward(
            frame,
            minimum_training_events=args.minimum_training_events,
            config=config,
        )
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    output = config.processed_data_dir / "baseline_evaluation.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["overall"], indent=2))
    print(f"Full walk-forward report written to {output}")


if __name__ == "__main__":
    main()
