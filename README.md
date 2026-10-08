# F1 Predictor

A reproducible Formula 1 data, modeling, and simulation project. It downloads
real historical data from Jolpica's Ergast-compatible API, builds
pre-race features, evaluates models chronologically, and provides a Streamlit
dashboard for walk-forward race predictions and clearly labeled scenarios.
Predictions and simulations are estimates, not guarantees or official F1
forecasts. No historical data or model metrics are bundled or fabricated.

## Architecture

```text
Jolpica API
    -> src/data/ (raw-response cache, canonical parsing, validation)
    -> data/processed/ (JSON Lines tables and manifest)
    -> src/features/ (chronological pre-race feature generation)
    -> src/models/ (baseline, XGBoost, walk-forward evaluation)
    -> src/simulation/ + src/scoring/ (race order, points, Monte Carlo)
    -> app/ (Streamlit presentation)
```

Processing is organized into independent packages. `F1DataProvider` is the
ingestion boundary; another provider can implement the same fetch methods
without changing the downstream canonical pipeline. Source responses are
cached under `data/raw/`; canonical records and run outputs are written under
`data/processed/`. These generated directories are excluded from Git.

## Requirements and installation

Python 3.10 or newer is required. Python 3.13 was used for project validation.
Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project does not require API credentials. Runtime settings can be
overridden with the `F1_*` environment variables in `src/utils/config.py`,
including API URL, raw and processed data directories, timeout, retry policy,
page size, selected season, random seed, and simulation count.

## Historical data ingestion

Download one season or an inclusive season range:

```powershell
python -m src.data.pipeline --start-season 2010
python -m src.data.pipeline --start-season 2018 --end-season 2024
```

If `--end-season` is omitted, the latest season advertised by the provider is
used. The client follows pagination, caches each endpoint and parameter set,
uses a configurable timeout, retries transient connection/server failures,
and observes `Retry-After` for HTTP 429/5xx responses. Invalid response
envelopes and invalid canonical records raise explicit errors.

The raw cache retains the parsed JSON response structure. It is not a
byte-for-byte archive of HTTP response formatting. Processed JSONL files are
normalized, validated records:

- `seasons.jsonl`, `races.jsonl`, `circuits.jsonl`
- `drivers.jsonl`, `constructors.jsonl`
- `qualifying_results.jsonl`, `race_results.jsonl`
- `driver_standings.jsonl`, `constructor_standings.jsonl`
- `manifest.json` with source, coverage, and record counts

The client can bypass its existing cache when used directly:

```python
from src.data.ingestion import JolpicaClient

client = JolpicaClient(refresh_cache=True)
```

The pipeline checks canonical field constraints, duplicate IDs, cross-table
race/entity references, season/round consistency, and impossible or duplicate
classified finishing positions. Data fixtures in the tests are mocked; tests
do not depend on the public API being available.

## Feature engineering

Build the processed historical feature matrix:

```powershell
python -m src.features.build
```

The default output is `data/processed/features.csv`; pass `--output path.csv`
to choose another project-relative path. Each row describes a driver at a
particular race entry. Rolling finishes, qualifying history, DNF rates,
circuit form, constructor form, and championship standings are computed from
strictly earlier events. A race's target finishing position, points, status,
and laps are outputs only; they are not used to calculate that row's history.

Current-event grid and qualifying positions are included because the
walk-forward race model is intended to represent a prediction made after
qualifying. Before-qualifying predictions must omit those two current-event
features. Standings features use the latest snapshot from an earlier event;
the target event's end-of-race standings are never used for its features.
`tests/test_features.py` checks that changing target and future outcomes does
not change the target's pre-race features.

Feature fields and their availability are defined by `FEATURE_COLUMNS` in
`src/features/engineering.py`. Driver/team/circuit IDs are categorical model
inputs. Position and DNF are the current supervised targets.

## Baselines, model training, and evaluation

Run previous-result, rolling-average, and Random Forest baselines using an
expanding race-by-race chronological validation:

```powershell
python -m src.models.evaluate
```

Evaluate XGBoost position, points, and DNF predictions as well:

```powershell
python -m src.models.evaluate --include-xgboost --minimum-training-events 5
```

Position and points regression are evaluated with MAE and RMSE. DNF classification reports
Brier score, log loss, and ROC-AUC when both classes are present. Raw and
sigmoid-calibrated DNF probability metrics are reported separately when the
chronological training history supports calibration. The evaluation command
writes actual fold results and metrics to
`data/processed/baseline_evaluation.json`; it does not invent values.

Train final models on the currently processed data:

```powershell
python -m src.models.train
python -m src.models.train --random-seed 17
```

Position and points estimators are XGBoost regressors with median imputation
and one-hot encoding. A DNF XGBoost classifier is trained when the data contains
both classes and enough chronological events; its sigmoid calibrator uses
expanding time-series out-of-fold predictions. XGBoost parameters are
centralized through `F1_RANDOM_SEED` and `AppConfig.model_parameters`.
Generated joblib models and model metadata go under `models/` and are ignored
by Git.

## Race and championship simulation

The race simulator uses sequential Plackett-Luce sampling from positive
relative driver strengths. DNF status is sampled from per-driver probabilities;
sampled DNFs are placed after finishers and randomized among themselves because
lap-completion detail is not modeled. They retain a complete order position
but have no classified position and score no points.

The separate scoring engine supplies historical season-specific race/sprint
tables and fastest-lap eligibility. Custom rules can be loaded from JSON using
`ScoringEngine.from_file(...)` or passed through `ScoringConfig`.

The championship Monte Carlo API is reusable from Python:

```python
from src.simulation.championship import ChampionshipSimulator, RaceForecast

forecast = RaceForecast(
    season=2024,
    round=10,
    strengths={"driver-a": 1.0, "driver-b": 0.8},
    dnf_probabilities={"driver-a": 0.05, "driver-b": 0.10},
)
result = ChampionshipSimulator().simulate(
    current_driver_points={"driver-a": 100, "driver-b": 90},
    remaining_races=[forecast],
    num_simulations=1000,
    random_seed=42,
)
```

Results include winner and finishing-position probabilities, expected and
median points, and expected finishing positions. Seeds make each run
reproducible. The scoring configuration can be changed independently from
model code.

## Streamlit dashboard

Launch from the repository root:

```powershell
python -m streamlit run app/streamlit_app.py
```

Pages include:

1. **Overview** — project, data coverage, model, and evaluation status.
2. **Race prediction** — chronological out-of-sample position estimates and an
   illustrative Plackett-Luce finishing distribution.
3. **Championship simulation** — historical cutoff standings, remaining
   schedule, and Monte Carlo summaries.
4. **What-if scenarios** — explicit hypothetical performance/DNF adjustments.

The dashboard explains missing prerequisites instead of substituting fake
results. Championship simulations use the selected completed race's
walk-forward performance estimate as a constant assumption for later scheduled
races; future qualifying, circuit-specific strengths, fastest laps, and sprint
outcomes are not known at that cutoff and are not fabricated.

## Testing

```powershell
python -m pytest
```

Tests cover schemas, API parsing and caching, malformed responses and retries,
data validation/persistence, leakage invariants, chronological model
evaluation, calibrated probabilities, race-order validity, scoring, and
seeded championship simulation. The API is mocked for offline tests.

## Limitations and future work

- Jolpica is a public external service; availability and rate limits are not
  controlled by this project.
- The raw cache is parsed JSON, not a signed or byte-exact API archive.
- Race position regression does not itself produce a probability
  distribution. Dashboard race-order probabilities are generated from a
  separate Plackett-Luce simulation using inverse expected-position strengths.
- DNF probability calibration is skipped if the chronological history is too
  small or lacks both classes; the dashboard explicitly states when its race
  distribution assumes no DNFs.
- The current data pipeline does not ingest sprint-result history or build a
  future circuit-specific forecast. Sprint scoring rules are available for
  configured simulation inputs.
- Championship scenarios hold a cutoff-race performance estimate constant
  across later rounds. They are scenario analyses, not guarantees.
- Historical scoring defaults cover common era-based points tables, including
  fastest-lap changes; unusual or shared-fastest-lap season rules should be
  explicitly configured. Historical dropped-score rules and shared-driver
  classifications are not currently applied to championship totals.
- No trained model or historical dataset is committed. Model quality must be
  measured on locally downloaded data before drawing conclusions.
