"""Race-level walk-forward prediction and ranking simulation page."""

import plotly.express as px
import streamlit as st

from app.ui_helpers import (
    get_event_fold,
    load_evaluation_report,
    load_processed_table,
    simulate_race_distribution,
)

st.subheader("Race prediction")
st.caption(
    "These are chronological walk-forward predictions for completed races: each "
    "model fold was trained only on earlier race events. A Plackett–Luce-style "
    "simulation turns position estimates into an illustrative ranking distribution; "
    "it is not a calibrated probability model for finishing positions."
)

try:
    report = load_evaluation_report()
    races = load_processed_table("races")
except (OSError, ValueError) as error:
    st.error(f"Could not load prediction inputs: {error}")
    st.stop()

if not report or not report.get("xgboost_position"):
    st.info(
        "No walk-forward predictions are available. Run "
        "`python -m src.models.evaluate --include-xgboost` after downloading "
        "historical data with the ingestion command."
    )
    st.stop()

folds = report["xgboost_position"].get("folds", [])
seasons = sorted({fold["season"] for fold in folds})
selected_season = st.selectbox("Season", seasons, index=len(seasons) - 1)
season_rounds = sorted(
    fold["round"] for fold in folds if fold["season"] == selected_season
)
selected_round = st.selectbox("Race", season_rounds, format_func=lambda round_number: (
    next(
        (
            race["name"]
            for race in races
            if race["season"] == selected_season and race["round"] == round_number
        ),
        f"Round {round_number}",
    )
))
fold = get_event_fold(report, "xgboost_position", selected_season, selected_round)
if fold is None:
    st.warning("No model evaluation fold exists for this race.")
    st.stop()

dnf_fold = get_event_fold(
    report, "xgboost_dnf_probability", selected_season, selected_round
)
dnf_probabilities = {
    row["driver_id"]: float(row["calibrated_dnf_probability"])
    for row in (dnf_fold or {}).get("predictions", [])
    if "calibrated_dnf_probability" in row
}
predictions = fold.get("predictions", [])
if not predictions:
    st.info("The selected race has no out-of-sample predictions.")
    st.stop()
if not dnf_probabilities:
    st.info(
        "No calibrated DNF probabilities were available for this fold; the "
        "illustrative ranking simulation assumes zero DNF probability."
    )

with st.container(horizontal=True):
    st.metric("Walk-forward MAE", f"{fold['metrics']['mae']:.2f} places", border=True)
    st.metric("Walk-forward RMSE", f"{fold['metrics']['rmse']:.2f} places", border=True)
    st.metric("Drivers evaluated", len(predictions), border=True)

with st.form("race_simulation_form"):
    simulations = st.number_input(
        "Ranking simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    submitted = st.form_submit_button("Generate finishing distribution")

if submitted:
    position_inputs = tuple(
        sorted(
            (row["driver_id"], float(row["expected_position"]))
            for row in predictions
        )
    )
    dnf_inputs = tuple(sorted(dnf_probabilities.items()))
    try:
        distribution = simulate_race_distribution(
            position_inputs,
            dnf_inputs,
            int(simulations),
            int(seed),
        )
    except ValueError as error:
        st.error(f"Could not simulate this race: {error}")
        st.stop()
    st.session_state["race_distribution"] = distribution
    st.session_state["race_actuals"] = {
        row["driver_id"]: row["actual_position"] for row in predictions
    }
    st.session_state["race_run_settings"] = (
        int(selected_season),
        int(selected_round),
        int(simulations),
        int(seed),
    )

distribution = st.session_state.get("race_distribution")
if (
    distribution is not None
    and st.session_state.get("race_run_settings")
    == (
        int(selected_season),
        int(selected_round),
        int(simulations),
        int(seed),
    )
):
    distribution = distribution.copy()
    actuals = st.session_state.get("race_actuals", {})
    distribution["actual_position"] = distribution["driver_id"].map(actuals)
    chart = px.bar(
        distribution.sort_values("expected_position"),
        x="driver_id",
        y="expected_position",
        color="p1_probability",
        hover_data=["p1_probability", "dnf_probability", "actual_position"],
        labels={
            "driver_id": "Driver",
            "expected_position": "Expected simulated finishing order",
            "p1_probability": "Simulated P1 probability",
        },
        title="Out-of-sample race prediction and simulated order",
    )
    chart.update_layout(yaxis_autorange="reversed")
    st.plotly_chart(chart)
    st.dataframe(
        distribution.sort_values("expected_position"),
        hide_index=True,
    )
