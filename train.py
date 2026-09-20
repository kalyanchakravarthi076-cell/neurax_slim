"""
Train multi-horizon congestion forecasting models (memory-safe).
Targets used only as labels — no leakage.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_utils import make_features, FEATURES

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
D = BASE / "data"
M = BASE / "models"
M.mkdir(exist_ok=True)

print("Loading limited rows for constrained RAM...")
usecols_t = [
    "timestamp", "segment_id", "speed_kmh", "flow_vph", "occupancy_pct",
    "travel_time_min", "free_flow_time_min", "delay_min", "queue_length_veh",
    "congestion_index", "sensor_quality",
]
# Read first 200k rows then sample — avoids loading entire 1.8M file
traffic = pd.read_csv(D / "traffic_train.csv", parse_dates=["timestamp"], usecols=usecols_t, nrows=200_000)
print(f"  loaded {len(traffic):,} traffic rows")
traffic = traffic.sample(min(60_000, len(traffic)), random_state=42)
print(f"  sample size {len(traffic):,}")

targets = pd.read_csv(
    D / "forecast_targets_train.csv",
    parse_dates=["timestamp"],
    usecols=["timestamp", "segment_id", "target_congestion_15m", "target_congestion_30m",
             "target_congestion_45m", "target_congestion_60m"],
    nrows=250_000,
)
d = traffic.merge(targets, on=["timestamp", "segment_id"], how="inner")
del traffic, targets
print(f"Merged rows: {len(d):,}")

net = pd.read_csv(D / "network.csv", usecols=["segment_id", "capacity_vph", "free_flow_speed_kmh"])
d = d.merge(net, on="segment_id", how="left")

ctx = None
if (D / "context_train.csv").exists():
    ctx = pd.read_csv(D / "context_train.csv", parse_dates=["timestamp"])

print("Building features...")
X = make_features(d, context=ctx)[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

models = {}
metrics = {}
for h in [15, 30, 45, 60]:
    target = f"target_congestion_{h}m"
    mask = d[target].notna()
    y = d.loc[mask, target].clip(0, 1).values
    Xh = X.loc[mask].values
    n = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(0.85 * n)
    tr, va = idx[:split], idx[split:]

    model = HistGradientBoostingRegressor(
        max_iter=50,
        max_leaf_nodes=31,
        learning_rate=0.1,
        l2_regularization=0.2,
        min_samples_leaf=40,
        random_state=42,
    )
    model.fit(Xh[tr], y[tr])
    pred_va = model.predict(Xh[va])
    mae_va = mean_absolute_error(y[va], pred_va)
    rmse_va = float(np.sqrt(mean_squared_error(y[va], pred_va)))
    models[h] = model
    metrics[h] = {"val_mae": round(mae_va, 4), "val_rmse": round(rmse_va, 4)}
    print(f"  +{h:2d} min  val MAE={mae_va:.4f}  val RMSE={rmse_va:.4f}")

bundle = {
    "models": models,
    "features": FEATURES,
    "metrics": metrics,
    "version": "neurax_v2_enhanced",
}
joblib.dump(bundle, M / "traffic_forecaster.joblib")
print("Saved models/traffic_forecaster.joblib")
print("Done.")
