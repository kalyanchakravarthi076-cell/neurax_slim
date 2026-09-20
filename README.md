# NeuraX Urban Traffic Intelligence — Enhanced Decision-Support System

Software-only AI decision-support prototype for large, rapidly changing urban road networks (Hyderabad-like: dense mixed traffic, peak commuter flows, signalised junctions, flyovers, recurring bottlenecks, roadworks, weather, events and incidents).

**This is not a navigation app or generic chatbot.** It produces a continuously updated view of network conditions, detects congestion and abnormal behaviour, classifies incident likelihood where data supports it, forecasts traffic states 15–60 minutes ahead, generates evidence-based operational/diversion advisories, and proposes data-driven infrastructure or network modifications for recurring bottlenecks together with estimated before/after impact.

All actions, diversion plans and construction suggestions remain **simulated / advisory**. No live signal control, camera access, GPS-device integration, roadside sensor integration, municipal infrastructure access or actual construction is performed or required.

---

## Scoring-aligned capabilities

| Criterion | How the system addresses it |
|-----------|-----------------------------|
| **Congestion & incident detection accuracy** | Transparent severity thresholds + multi-signal anomaly score + heuristic incident likelihood requiring multiple stress indicators (CI, queue, occupancy, speed, delay). Sensor quality down-weights unreliable readings to control false alarms. Spillback detection via neighbour graph. |
| **Traffic forecasting accuracy** | Multi-horizon (15/30/45/60 min) HistGradientBoosting models with enriched features (physical ratios, peak flag, capacity utilisation, weather/event context). Targets used only as labels — no leakage. Validation MAE reported in model bundle. |
| **Quality of adaptive recommendations** | Priority-tiered advisories (Critical / High / Elevated / Normal) with concrete actions, preferred alternate corridors from live neighbour load, expected impact language, and full evidence chain. |
| **Robustness to unseen patterns** | Explicit sensor-quality feature, context (rain, events, holidays), what-if stress testing UI, graceful degradation when model file is absent, spillback awareness. |
| **Explainability & confidence** | Fixed severity thresholds, published anomaly formula, incident evidence strings, forecast confidence derived from sensor quality, limitations section, feature list, held-out metrics. |
| **Technical implementation** | Modular `model_utils` / `train` / `app`, cached data loaders, reproducible training script, joblib model bundle with version & metrics. |
| **UI/UX & visualisation** | Wide Streamlit layout, interactive map (severity-coloured links, incident & roadwork overlays), time slider across validation window, forecast charts with thresholds, tables, side-bar controls, six focused tabs. |
| **Innovation** | Neighbour-aware spillback detection, planning-candidate matching for recurring bottlenecks with transparent impact heuristic, context-aware features, what-if advisory stress test, multi-signal incident scoring that stays fully observable and auditable. |

---

## Features (enhanced over Checkpoint 2)

- Network congestion dashboard with geospatial road visualisation and severity colouring
- Time-slider navigation of the validation timeline
- Incident & roadwork overlays on the map
- Transparent congestion severity classification
- Multi-signal anomaly / stress ranking
- Heuristic incident likelihood with written evidence
- Spillback risk detection across neighbouring segments
- ML forecasts for congestion at +15 / +30 / +45 / +60 minutes (context-aware features)
- Evidence-based operational diversion & traffic-management advisories with priority, actions, confidence and neighbour alternatives
- Data-driven infrastructure suggestions matched to `planning_candidates.csv` with estimated CI reduction
- Explainability panel (features, formulas, limitations, validation metrics)
- What-if robustness / stress-test panel
- Full use of organiser context, incidents, roadworks and planning candidate files

---

## Quick start

```bash
# 1. Create / activate a Python environment (3.10+)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Organiser CSV files are already in data/

# 4. Train forecasting models (samples ~280k rows for speed; ~1–3 min on a laptop)
python train.py

# 5. Launch the decision-support UI
streamlit run app.py
```

---

## Important modelling notes

- `forecast_targets_*.csv` columns are used **only as supervised labels** in `train.py`. They are never present in the feature matrix, satisfying the no-leakage requirement.
- Feature engineering lives in `model_utils.make_features` and can optionally merge weather / event context.
- Anomaly and incident scores are pure functions of observables; they do not depend on the regression targets.
- Infrastructure impact numbers are transparent heuristics for ranking candidates, not microsimulation outputs.

---

## Demo flow (recommended for judges)

1. **Network Map** — inspect severe corridors, toggle incident / roadwork overlays, move the time slider.
2. **Forecast & Advisory** — select a high-CI segment → view 15–60 min trajectory → read priority, actions and evidence.
3. **Detection & Incidents** — review anomaly ranking, incident likelihood table, spillback clusters and labelled incidents active at the snapshot.
4. **Infrastructure Planning** — examine recurring-bottleneck matches to planning candidates and estimated before/after CI.
5. **Explainability** — review formulas, features, confidence logic and limitations.
6. **Robustness** — stress-test a segment’s observables and observe how advisory priority adapts.

---

## Project layout

```
neurax_final/
├── app.py              # Streamlit decision-support UI
├── train.py            # Model training (multi-horizon HGBR)
├── model_utils.py      # Features, severity, anomaly, incident, spillback helpers
├── requirements.txt
├── README.md
├── models/             # traffic_forecaster.joblib (created by train.py)
└── data/               # Organiser CSVs (train + validation)
```

---

## Disclaimer

This prototype is **advisory and simulated only**. It does not control real signals, cameras, GPS fleets or municipal infrastructure. Diversion and construction suggestions are decision-support artefacts for evaluation of the AI system, not operational commands.

---

## Note on this slim package

This archive is a **run-ready** package (~12 MB) that includes:

- All application code (`app.py`, `model_utils.py`, `train.py`)
- Pre-trained forecasting model
- Validation data + network/nodes/context/incidents/roadworks/planning files needed by the UI

It **excludes** the very large training CSVs (`traffic_train.csv`, `forecast_targets_train.csv`) so the download stays small and reliable.  
The app runs fully with the included model. Re-training is optional and only needed if you want to rebuild the model from scratch (in that case use the full original data from the organiser).
