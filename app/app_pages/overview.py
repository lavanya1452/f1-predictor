"""Project and data coverage overview page."""

import streamlit as st

from app.ui_helpers import load_evaluation_report, load_manifest, load_processed_table
from src.utils.config import AppConfig

st.subheader("A data-grounded F1 forecasting workbench")
st.write(
    "Explore historical race data, leakage-aware walk-forward model results, "
    "and seeded race and championship simulations. Simulations are estimates, "
    "not guarantees or official Formula 1 predictions."
)

try:
    races = load_processed_table("races")
    manifest = load_manifest()
except (OSError, ValueError) as error:
    st.error(f"Could not load the local data foundation: {error}")
    races = []
    manifest = None

config = AppConfig.from_env()
counts = (manifest or {}).get("record_counts", {})
with st.container(horizontal=True):
    st.metric(
        "Selected season",
        config.selected_season if config.selected_season is not None else "Not set",
        border=True,
    )
    st.metric("Seasons", len({race["season"] for race in races}), border=True)
    st.metric("Race records", counts.get("races", len(races)), border=True)
    st.metric(
        "Processed tables",
        len(counts),
        border=True,
    )

with st.container(border=True):
    st.subheader("Project status")
    if not races:
        st.info(
            "No processed historical data is available yet. Download real data with "
            "`python -m src.data.pipeline --start-season 2010`."
        )
    else:
        seasons = sorted({race["season"] for race in races})
        st.write(f"Coverage: {seasons[0]}–{seasons[-1]}")
    metadata_file = config.model_dir / "model_metadata.json"
    model_file = config.model_dir / "position_xgboost.joblib"
    st.write(
        "Position model: **trained**"
        if model_file.is_file()
        else "Position model: **not trained**"
    )
    if metadata_file.is_file():
        st.caption(f"Model metadata: `{metadata_file}`")
    evaluation = load_evaluation_report()
    if evaluation:
        model_report = evaluation.get("xgboost_position", {})
        st.write(
            f"Walk-forward position evaluation: "
            f"{model_report.get('fold_count', 0)} race folds"
        )
    else:
        st.info(
            "Walk-forward predictions are not available. After downloading data, run "
            "`python -m src.models.evaluate --include-xgboost`."
        )
    st.caption(f"Random seed configured for training: {config.random_seed}")
    st.caption(
        "Last processed data update: "
        + str((manifest or {}).get("updated_at", "not available"))
    )

st.caption(
    "The app works without a trained model: pages explain which real-data steps "
    "are needed and do not fill gaps with fabricated predictions."
)
