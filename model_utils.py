"""
NeuraX model utilities — enhanced feature engineering for congestion forecasting,
incident scoring, and explainable decision support.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Core observable features used by the forecaster (no target leakage)
FEATURES = [
    "speed_kmh",
    "flow_vph",
    "occupancy_pct",
    "travel_time_min",
    "free_flow_time_min",
    "delay_min",
    "queue_length_veh",
    "congestion_index",
    "sensor_quality",
    "hour",
    "minute",
    "dow",
    "is_peak",
    "speed_ratio",
    "capacity_util",
    "queue_pressure",
    "delay_ratio",
    "temp_c",
    "rain",
    "event_level",
    "holiday_flag",
]


def make_features(df: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build model-ready features from a traffic snapshot (and optional context)."""
    x = df.copy()
    if "timestamp" in x.columns:
        ts = pd.to_datetime(x["timestamp"])
        x["hour"] = ts.dt.hour
        x["minute"] = ts.dt.minute
        x["dow"] = ts.dt.dayofweek
    else:
        x["hour"] = x.get("hour", 0)
        x["minute"] = x.get("minute", 0)
        x["dow"] = x.get("dow", 0)

    # Peak-hour indicator (Hyderabad-like morning/evening peaks)
    x["is_peak"] = (
        ((x["hour"] >= 7) & (x["hour"] <= 10)) | ((x["hour"] >= 17) & (x["hour"] <= 21))
    ).astype(int)

    # Derived physical ratios (robust to missing free-flow)
    fft = x.get("free_flow_time_min", pd.Series(1.0, index=x.index)).replace(0, np.nan)
    free_spd = x.get("free_flow_speed_kmh", pd.Series(40.0, index=x.index)).replace(0, np.nan)
    x["speed_ratio"] = (x.get("speed_kmh", 0) / free_spd).fillna(1.0)
    x["delay_ratio"] = (x.get("delay_min", 0) / fft).fillna(0).clip(0, 10)
    x["queue_pressure"] = np.clip(x.get("queue_length_veh", 0) / 40.0, 0, 3)

    # Capacity utilisation if capacity available
    if "capacity_vph" in x.columns:
        x["capacity_util"] = (x.get("flow_vph", 0) / x["capacity_vph"].replace(0, np.nan)).fillna(0).clip(0, 2)
    else:
        x["capacity_util"] = np.clip(x.get("occupancy_pct", 0) / 100.0, 0, 1.5)

    # Context (weather / events) — left-join by timestamp if provided
    x["temp_c"] = 25.0
    x["rain"] = 0.0
    x["event_level"] = 0
    x["holiday_flag"] = 0
    if context is not None and "timestamp" in x.columns and "timestamp" in context.columns:
        ctx = context.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
        x["_ts5"] = pd.to_datetime(x["timestamp"]).dt.floor("5min")
        ctx["_ts5"] = ctx["timestamp"].dt.floor("5min")
        cols = [c for c in ["temperature_c", "rain_intensity", "event_level", "holiday_flag"] if c in ctx.columns]
        if cols:
            merged = x.merge(ctx[["_ts5"] + cols].drop_duplicates("_ts5"), on="_ts5", how="left")
            x["temp_c"] = merged.get("temperature_c", pd.Series(25.0, index=x.index)).fillna(25.0)
            x["rain"] = merged.get("rain_intensity", pd.Series(0.0, index=x.index)).fillna(0.0)
            x["event_level"] = merged.get("event_level", pd.Series(0, index=x.index)).fillna(0)
            x["holiday_flag"] = merged.get("holiday_flag", pd.Series(0, index=x.index)).fillna(0)
        x = x.drop(columns=["_ts5"], errors="ignore")

    # Ensure all FEATURES exist
    for f in FEATURES:
        if f not in x.columns:
            x[f] = 0.0
    return x


def anomaly_score(row_or_df) -> pd.Series | float:
    """Transparent, multi-signal anomaly / stress score in [0, 1+]."""
    if isinstance(row_or_df, pd.DataFrame):
        t = row_or_df
        return (
            0.40 * t["congestion_index"].clip(0, 1)
            + 0.20 * np.clip(t["occupancy_pct"] / 100.0, 0, 1)
            + 0.20 * np.clip(t["queue_length_veh"] / 50.0, 0, 1)
            + 0.10 * np.clip(t.get("delay_min", 0) / 8.0, 0, 1)
            + 0.10 * (1.0 - t.get("sensor_quality", 1.0).clip(0, 1))
        )
    r = row_or_df
    return float(
        0.40 * min(max(r.congestion_index, 0), 1)
        + 0.20 * min(max(r.occupancy_pct / 100.0, 0), 1)
        + 0.20 * min(max(r.queue_length_veh / 50.0, 0), 1)
        + 0.10 * min(max(getattr(r, "delay_min", 0) / 8.0, 0), 1)
        + 0.10 * (1.0 - min(max(getattr(r, "sensor_quality", 1.0), 0), 1))
    )


def severity(ci: float) -> str:
    if ci >= 0.65:
        return "Severe"
    if ci >= 0.40:
        return "High"
    if ci >= 0.20:
        return "Moderate"
    return "Normal"


def severity_color(sev: str) -> str:
    return {
        "Normal": "#22c55e",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Severe": "#ef4444",
    }.get(sev, "#94a3b8")


def confidence_from_sensor(sensor_quality: float, n_obs: int = 1) -> float:
    """Simple confidence score combining sensor quality and observation support."""
    base = float(np.clip(sensor_quality, 0.3, 1.0))
    support = min(1.0, 0.6 + 0.4 * min(n_obs, 5) / 5)
    return round(base * support, 3)


def classify_incident_likelihood(row) -> tuple[str, float, str]:
    """
    Heuristic incident likelihood from observable stress signals.
    Returns (label, score, evidence).
    """
    ci = float(row.congestion_index)
    queue = float(row.queue_length_veh)
    occ = float(row.occupancy_pct)
    speed = float(row.speed_kmh)
    delay = float(getattr(row, "delay_min", 0))
    sq = float(getattr(row, "sensor_quality", 1.0))

    score = 0.0
    evidence = []
    if ci >= 0.7 and queue >= 25:
        score += 0.45
        evidence.append(f"very high congestion ({ci:.0%}) + long queue ({queue:.0f} veh)")
    elif ci >= 0.5 and queue >= 15:
        score += 0.30
        evidence.append(f"elevated congestion ({ci:.0%}) + queue ({queue:.0f})")
    if occ >= 70 and speed < 15:
        score += 0.25
        evidence.append(f"high occupancy ({occ:.0f}%) with low speed ({speed:.1f} km/h)")
    if delay >= 5:
        score += 0.15
        evidence.append(f"significant delay ({delay:.1f} min)")
    if sq < 0.7:
        score *= 0.85
        evidence.append("reduced sensor quality → lower confidence")

    score = min(1.0, score)
    if score >= 0.65:
        label = "Likely incident / blockage"
    elif score >= 0.40:
        label = "Possible incident / surge"
    elif score >= 0.25:
        label = "Elevated stress (monitor)"
    else:
        label = "Normal operations"
    return label, round(score, 3), "; ".join(evidence) if evidence else "No strong abnormal signals"


def build_neighbor_map(network: pd.DataFrame) -> dict[str, list[str]]:
    """Undirected adjacency of segments via shared nodes (for spillback checks)."""
    from collections import defaultdict
    node_to_segs: dict = defaultdict(list)
    for _, r in network.iterrows():
        node_to_segs[r["source_node"]].append(r["segment_id"])
        node_to_segs[r["target_node"]].append(r["segment_id"])
    neighbors: dict[str, set] = defaultdict(set)
    for segs in node_to_segs.values():
        for s in segs:
            for t in segs:
                if s != t:
                    neighbors[s].add(t)
    return {k: sorted(v) for k, v in neighbors.items()}


def detect_spillback(snap: pd.DataFrame, neighbors: dict[str, list[str]], threshold: float = 0.45) -> pd.DataFrame:
    """Flag segments whose congestion is accompanied by congested neighbors (spillback risk)."""
    ci_map = dict(zip(snap["segment_id"], snap["congestion_index"]))
    rows = []
    for _, r in snap.iterrows():
        segs = neighbors.get(r["segment_id"], [])
        neigh_ci = [ci_map[s] for s in segs if s in ci_map]
        if not neigh_ci:
            continue
        max_n = max(neigh_ci)
        mean_n = float(np.mean(neigh_ci))
        if r["congestion_index"] >= threshold and mean_n >= threshold * 0.85:
            rows.append(
                {
                    "segment_id": r["segment_id"],
                    "congestion_index": r["congestion_index"],
                    "neighbor_mean_ci": round(mean_n, 3),
                    "neighbor_max_ci": round(max_n, 3),
                    "n_neighbors": len(neigh_ci),
                    "spillback_risk": "High" if mean_n >= 0.55 else "Moderate",
                }
            )
    return pd.DataFrame(rows).sort_values("congestion_index", ascending=False) if rows else pd.DataFrame()
