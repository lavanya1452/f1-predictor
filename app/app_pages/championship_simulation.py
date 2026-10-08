"""Monte Carlo championship simulation using historical walk-forward forecasts."""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_helpers import (
    load_evaluation_report,
    load_processed_table,
    prepare_simulation_context,
    run_championship_simulation,
    forecast_dicts,
)

st.subheader("Championship simulation")
st.caption(
    "Choose a historical cutoff after a completed race. The cutoff's walk-forward "
    "performance estimate is held constant across later scheduled races because "
    "future qualifying and circuit-specific forecasts are not available at that "
    "cutoff. This is a documented scenario assumption, not an official prediction."
)

try:
    report = load_evaluation_report()
    races = load_processed_table("races")
except (OSError, ValueError) as error:
    st.error(f"Could not load simulation inputs: {error}")
    st.stop()

if not report or not report.get("xgboost_position"):
    st.info(
        "Generate chronological model predictions first with "
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
    st.info(
        "No evaluated cutoff has remaining scheduled races in the downloaded data. "
        "Refresh the historical dataset when more schedule information is available."
    )
    st.stop()

selected_season = st.selectbox(
    "Season",
    sorted({event[0] for event in available_events}),
    index=len({event[0] for event in available_events}) - 1,
)
rounds = [round_number for season, round_number in available_events if season == selected_season]
selected_round = st.selectbox("Standings cutoff after round", rounds)

try:
    context, dnf_note = prepare_simulation_context(
        int(selected_season), int(selected_round), report
    )
except (OSError, ValueError) as error:
    st.warning(str(error))
    st.stop()

if dnf_note:
    st.info(dnf_note)
if not context["constructor_by_driver"]:
    st.warning(
        "No constructor mapping is available for the selected standings cutoff; "
        "constructor championship output may be incomplete."
    )

future_rounds = [forecast.round for forecast in context["forecasts"]]
st.caption(
    f"Simulating rounds {min(future_rounds)}–{max(future_rounds)} "
    f"({len(future_rounds)} scheduled events)."
)

with st.form("championship_simulation_form"):
    simulation_count = st.number_input(
        "Number of simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    submitted = st.form_submit_button("Run Monte Carlo simulation", type="primary")

if submitted:
    try:
        result = run_championship_simulation(
            context["driver_points"],
            context["constructor_points"],
            context["constructor_by_driver"],
            forecast_dicts(context["forecasts"]),
            int(simulation_count),
            int(random_seed),
        )
    except ValueError as error:
        st.error(f"Simulation input was invalid: {error}")
        st.stop()
    st.session_state["championship_result"] = result
    st.session_state["championship_settings"] = (
        int(selected_season),
        int(selected_round),
        int(simulation_count),
        int(random_seed),
    )

result = st.session_state.get("championship_result")
if (
    result
    and st.session_state.get("championship_settings")
    == (
        int(selected_season),
        int(selected_round),
        int(simulation_count),
        int(random_seed),
    )
):
    driver_rows = [
        {"driver_id": driver, **values}
        for driver, values in result["driver_championship"].items()
    ]
    driver_frame = pd.DataFrame(driver_rows).sort_values(
        ["winner_probability", "expected_final_position"],
        ascending=[False, True],
    )
    with st.container(horizontal=True):
        st.metric("Simulations", result["simulations"], border=True)
        st.metric(
            "Winner probability total",
            f"{sum(row['winner_probability'] for row in result['driver_championship'].values()):.3f}",
            border=True,
        )
        st.metric("Random seed", result["random_seed"], border=True)
    with st.container(border=True):
        st.subheader("Championship winner probabilities")
        winner_chart = px.bar(
            driver_frame,
            x="driver_id",
            y="winner_probability",
            title="Championship winner probability",
            labels={"driver_id": "Driver", "winner_probability": "Probability"},
        )
        st.plotly_chart(winner_chart)
    with st.container(border=True):
        st.subheader("Final position distribution")
        position_rows = [
            {
                "driver_id": row["driver_id"],
                "position": int(position),
                "probability": probability,
            }
            for row in driver_rows
            for position, probability in row["position_probabilities"].items()
        ]
        position_frame = pd.DataFrame(position_rows)
        chart = px.bar(
            position_frame,
            x="position",
            y="probability",
            color="driver_id",
            barmode="group",
            title="Probability of each final championship position",
        )
        st.plotly_chart(chart)
        st.dataframe(driver_frame, hide_index=True)
    constructor_results = result.get("constructor_championship")
    if constructor_results:
        st.subheader("Constructor championship")
        st.dataframe(
            pd.DataFrame(
                [
                    {"constructor_id": team, **values}
                    for team, values in constructor_results.items()
                ]
            ).sort_values("winner_probability", ascending=False),
            hide_index=True,
        )
