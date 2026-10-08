"""Interactive hypothetical championship scenario page."""

from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_helpers import (
    forecast_dicts,
    load_evaluation_report,
    load_processed_table,
    prepare_simulation_context,
    run_championship_simulation,
)

st.subheader("What-if scenarios")
st.caption(
    "Every adjustment on this page is hypothetical. It does not alter source data, "
    "trained models, or actual Formula 1 results."
)

try:
    report = load_evaluation_report()
    races = load_processed_table("races")
except (OSError, ValueError) as error:
    st.error(f"Could not load scenario inputs: {error}")
    st.stop()

if not report or not report.get("xgboost_position"):
    st.info(
        "Generate walk-forward predictions first with "
        "`python -m src.models.evaluate --include-xgboost`."
    )
    st.stop()

folds = report["xgboost_position"].get("folds", [])
available_events = sorted(
    {
        (fold["season"], fold["round"])
        for fold in folds
        if any(
            race["season"] == fold["season"] and race["round"] > fold["round"]
            for race in races
        )
    }
)
if not available_events:
    st.info("No evaluated cutoff has remaining scheduled races.")
    st.stop()

selected_season = st.selectbox(
    "Season",
    sorted({event[0] for event in available_events}),
    index=len({event[0] for event in available_events}) - 1,
)
selected_round = st.selectbox(
    "Standings cutoff after round",
    [round_number for season, round_number in available_events if season == selected_season],
)
try:
    context, dnf_note = prepare_simulation_context(
        int(selected_season), int(selected_round), report
    )
except (OSError, ValueError) as error:
    st.warning(str(error))
    st.stop()
if dnf_note:
    st.info(dnf_note)

driver_options = sorted(context["driver_points"])
selected_driver = st.selectbox("Driver to adjust", driver_options)
base_dnf = context["forecasts"][0].dnf_probabilities.get(selected_driver, 0.0)
with st.form("what_if_form"):
    performance_multiplier = st.slider(
        "Driver performance multiplier",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Scales the selected driver's relative Plackett–Luce strength.",
    )
    dnf_probability = st.slider(
        "Assumed DNF probability",
        min_value=0.0,
        max_value=1.0,
        value=float(base_dnf),
        step=0.01,
        help="Overrides the selected driver's DNF probability for each remaining race.",
    )
    simulation_count = st.number_input(
        "Number of simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    submitted = st.form_submit_button("Run hypothetical scenario", type="primary")

if submitted:
    forecasts = []
    for forecast in context["forecasts"]:
        strengths = dict(forecast.strengths)
        strengths[selected_driver] *= performance_multiplier
        dnf_values = dict(forecast.dnf_probabilities)
        dnf_values[selected_driver] = float(dnf_probability)
        forecasts.append(
            replace(
                forecast,
                strengths=strengths,
                dnf_probabilities=dnf_values,
            )
        )
    try:
        result = run_championship_simulation(
            context["driver_points"],
            context["constructor_points"],
            context["constructor_by_driver"],
            forecast_dicts(forecasts),
            int(simulation_count),
            int(random_seed),
        )
    except ValueError as error:
        st.error(f"Scenario inputs were invalid: {error}")
        st.stop()
    st.session_state["what_if_result"] = result
    st.session_state["what_if_settings"] = (
        int(selected_season),
        int(selected_round),
        selected_driver,
        float(performance_multiplier),
        float(dnf_probability),
        int(simulation_count),
        int(random_seed),
    )

result = st.session_state.get("what_if_result")
if (
    result
    and st.session_state.get("what_if_settings")
    == (
        int(selected_season),
        int(selected_round),
        selected_driver,
        float(performance_multiplier),
        float(dnf_probability),
        int(simulation_count),
        int(random_seed),
    )
):
    rows = [
        {"driver_id": driver, **values}
        for driver, values in result["driver_championship"].items()
    ]
    frame = pd.DataFrame(rows).sort_values(
        "winner_probability", ascending=False
    )
    st.metric(
        f"{selected_driver} championship win probability",
        f"{result['driver_championship'][selected_driver]['winner_probability']:.1%}",
        border=True,
    )
    chart = px.bar(
        frame,
        x="driver_id",
        y="winner_probability",
        title="Hypothetical championship winner probabilities",
        labels={"driver_id": "Driver", "winner_probability": "Probability"},
    )
    st.plotly_chart(chart)
    st.dataframe(frame, hide_index=True)
