"""Build the leakage-safe historical feature matrix."""

import argparse

from src.features.engineering import FeatureBuilder
from src.utils.config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-race F1 feature rows.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to data/processed/features.csv).",
    )
    args = parser.parse_args()
    config = AppConfig.from_env()
    frame = FeatureBuilder(config).build_from_processed()
    output = (
        config.processed_data_dir / "features.csv"
        if args.output is None
        else config.project_root / args.output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} feature rows to {output}")


if __name__ == "__main__":
    main()
