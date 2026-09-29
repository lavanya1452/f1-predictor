# F1 Predictor

An end-to-end system for exploring Formula 1 race and championship prediction.
Predictions and simulated probabilities are estimates, not guarantees.

## Stage 1: Data Foundation

The initial implementation provides validated canonical records and a client
for the public Jolpica API, which is compatible with Ergast-style endpoints.
Responses are retained unmodified as JSON files in `data/raw/`; repeated
requests use the local cache. No historical records are bundled or fabricated.

Canonical models in `src/data/schemas.py` cover circuits, drivers,
constructors, races, qualifying results, race results, driver standings, and
constructor standings. Records use `season`, `round`, `race_id`,
`driver_id`, `constructor_id`, and `circuit_id` consistently where applicable.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The client is opt-in; importing the package does not make network requests.
For example, from the repository root:

```powershell
python -c "from src.data import JolpicaClient; print(JolpicaClient().fetch_seasons()[:2])"
```

Configuration can be set with `F1_*` environment variables. Available
settings and defaults are defined in `src/utils/config.py`. Tests run with:

```powershell
pytest
```

Feature engineering, models, simulation, and the dashboard are planned stages
and are not implemented yet.