"""Train race prediction models from locally processed historical data."""

import argparse

from src.features.engineering import FeatureBuilder
from src.models.training import train_models
from src.utils.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F1 prediction models.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the configured model seed for this reproducible run.",
    )
    args = parser.parse_args()
    config = AppConfig.from_env()
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    frame = FeatureBuilder(config).build_from_processed()
    artifacts = train_models(frame, config)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
