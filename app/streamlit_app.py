"""Entry point for the F1 predictor dashboard."""

from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(
        page_title="F1 Predictor",
        layout="wide",
    )

    page = st.navigation(
        [
            st.Page(
                str(APP_DIR / "app_pages" / "overview.py"),
                title="Overview",
                icon=":material/dashboard:",
                default=True,
            ),
            st.Page(
                str(APP_DIR / "app_pages" / "race_prediction.py"),
                title="Race prediction",
                icon=":material/flag:",
            ),
            st.Page(
                str(APP_DIR / "app_pages" / "championship_simulation.py"),
                title="Championship simulation",
                icon=":material/leaderboard:",
            ),
            st.Page(
                str(APP_DIR / "app_pages" / "what_if.py"),
                title="What-if scenarios",
                icon=":material/tune:",
            ),
        ],
        position="top",
    )

    st.title(page.title, icon=page.icon)
    page.run()


if __name__ == "__main__":
    main()
