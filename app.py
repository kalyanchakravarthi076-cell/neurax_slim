"""
NeuraX Urban Traffic Intelligence — Enhanced Decision-Support System
Software-only advisory prototype for Hyderabad-like dense mixed traffic networks.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model_utils import (
    FEATURES,
    make_features,
    anomaly_score,
    severity,
    severity_color,
    confidence_from_sensor,
    classify_incident_likelihood,
    build_neighbor_map,
    detect_spillback,
)

st.set_page_config(
    page_title="NeuraX Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA = BASE / "data"
MODEL_PATH = BASE / "models" / "traffic_forecaster.joblib"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading network…")
def load_static():
    net = pd.read_csv(DATA / "network.csv")
    nodes = pd.read_csv(DATA / "nodes.csv")
    src = nodes.rename(columns={"node_id": "source_node", "lat": "src_lat", "lon": "src_lon"})[
        ["source_node", "src_lat", "src_lon"]
    ]
    dst = nodes.rename(columns={"node_id": "target_node", "lat": "dst_lat", "lon": "dst_lon"})[
        ["target_node", "dst_lat", "dst_lon"]
    ]
    net = net.merge(src, on="source_node").merge(dst, on="target_node")
    return net, nodes


@st.cache_data(show_spinner="Loading traffic validation…")
def load_traffic():
    return pd.read_csv(DATA / "traffic_validation.csv", parse_dates=["timestamp"])


@st.cache_data
def load_context():
    p = DATA / "context_validation.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    return None


@st.cache_data
def load_incidents():
    p = DATA / "incidents_validation.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["start_time", "end_time"])
    return pd.DataFrame()


@st.cache_data
def load_roadworks():
    p = DATA / "roadworks_validation.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["start_time", "end_time"])
    return pd.DataFrame()


@st.cache_data
def load_planning():
    p = DATA / "planning_candidates.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# ---------------------------------------------------------------------------
# Advisory engine
# ---------------------------------------------------------------------------
def generate_advisory(row, preds: dict, net: pd.DataFrame, neighbors: dict, snap: pd.DataFrame) -> dict:
    """Produce evidence-based operational advisory with reasoning and expected impact."""
    seg = row.segment_id
    current = float(row.congestion_index)
    p15, p30, p60 = preds.get(15, current), preds.get(30, current), preds.get(60, current)
    label, inc_score, inc_ev = classify_incident_likelihood(row)
    conf = confidence_from_sensor(float(row.sensor_quality))

    # Neighbor load
    neigh = neighbors.get(seg, [])
    neigh_ci = snap.set_index("segment_id")["congestion_index"].reindex(neigh).dropna()
    avg_neigh = float(neigh_ci.mean()) if len(neigh_ci) else 0.0
    free_neigh = [s for s in neigh if s in snap.segment_id.values and snap.loc[snap.segment_id == s, "congestion_index"].iloc[0] < 0.30]

    actions = []
    impact = "Low"
    priority = "Routine"

    if p30 >= 0.70 or (current >= 0.65 and p15 >= 0.60):
        priority = "Critical"
        impact = "High — expected queue growth and spillback risk"
        actions.append(f"Immediate diversion of inbound traffic away from {seg}.")
        if free_neigh:
            actions.append(f"Preferred alternate corridors (lower load): {', '.join(free_neigh[:3])}.")
        else:
            actions.append("No clearly free neighbor; coordinate multi-segment diversion and temporary inflow metering.")
        if inc_score >= 0.4:
            actions.append(f"Treat as possible incident ({label}, score {inc_score:.2f}): dispatch verification and clear blockage if confirmed.")
        actions.append("Pre-position traffic marshals / temporary signage at upstream junctions.")
    elif p30 >= 0.45 or (p30 > current + 0.12):
        priority = "High"
        impact = "Moderate — deterioration expected within 30 min"
        actions.append(f"Prepare diversion readiness for {seg}; monitor every 5 min.")
        if free_neigh:
            actions.append(f"Stand-by alternate routes: {', '.join(free_neigh[:3])}.")
        actions.append("Consider signal timing adjustment (green extension on exit movements) if plan allows.")
        if avg_neigh > 0.40:
            actions.append("Neighbor segments already stressed — watch for spillback.")
    elif current >= 0.35:
        priority = "Elevated"
        impact = "Low–Moderate — stable but elevated load"
        actions.append(f"Continue active monitoring of {seg}; no hard diversion yet.")
        actions.append("Share advisory with operators of adjacent corridors.")
    else:
        priority = "Normal"
        impact = "Minimal"
        actions.append(f"{seg} operating within normal bounds; routine surveillance sufficient.")

    reasoning = (
        f"Current CI={current:.0%}, forecast +15m={p15:.0%}, +30m={p30:.0%}, +60m={p60:.0%}. "
        f"Queue={row.queue_length_veh:.0f} veh, occupancy={row.occupancy_pct:.0f}%, "
        f"speed={row.speed_kmh:.1f} km/h. Neighbor avg CI={avg_neigh:.0%}. "
        f"Incident likelihood: {label} ({inc_score:.2f}). Confidence={conf:.0%}."
    )
    return {
        "priority": priority,
        "impact": impact,
        "actions": actions,
        "reasoning": reasoning,
        "confidence": conf,
        "incident_label": label,
        "incident_score": inc_score,
        "incident_evidence": inc_ev,
        "free_neighbors": free_neigh[:5],
    }


def infrastructure_suggestions(snap: pd.DataFrame, planning: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Match recurring high-congestion segments to planning candidates and estimate impact."""
    if planning.empty:
        return pd.DataFrame()
    # Recurring load: average CI over validation window
    rec = (
        traffic.groupby("segment_id")["congestion_index"]
        .agg(["mean", "max", "count"])
        .rename(columns={"mean": "avg_ci", "max": "max_ci", "count": "n_obs"})
        .reset_index()
    )
    rec = rec[rec["avg_ci"] >= 0.28].sort_values("avg_ci", ascending=False)
    # Join candidates (target_segment)
    cand = planning.copy()
    if "target_segment" not in cand.columns:
        return pd.DataFrame()
    merged = cand.merge(rec, left_on="target_segment", right_on="segment_id", how="inner")
    if merged.empty:
        # also try segment_id column if present
        if "segment_id" in cand.columns:
            merged = cand.merge(rec, on="segment_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    def est_impact(row):
        # Simple expected reduction proportional to capacity delta and current load
        delta = float(row.get("capacity_delta_vph", 300))
        base = float(row["avg_ci"])
        # heuristic: larger capacity gain + higher current CI → larger relative relief
        relief = min(0.45, 0.08 + 0.00025 * delta + 0.25 * base)
        return round(relief, 3)

    merged["est_ci_reduction"] = merged.apply(est_impact, axis=1)
    merged["est_after_ci"] = (merged["avg_ci"] - merged["est_ci_reduction"]).clip(lower=0.05)
    merged["priority_score"] = (
        merged["avg_ci"] * 0.5
        + merged["est_ci_reduction"] * 0.3
        + (1.0 / (merged.get("cost_index", 10) + 1)) * 0.2
    )
    cols = [
        "candidate_id",
        "target_segment",
        "intervention_type",
        "capacity_delta_vph",
        "cost_index",
        "feasibility_band",
        "avg_ci",
        "max_ci",
        "est_ci_reduction",
        "est_after_ci",
        "priority_score",
    ]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values("priority_score", ascending=False).head(25)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
net, nodes = load_static()
traffic = load_traffic()
context = load_context()
incidents = load_incidents()
roadworks = load_roadworks()
planning = load_planning()
model_bundle = load_model()
neighbors = build_neighbor_map(net)

st.title("🚦 NeuraX Urban Traffic Intelligence")
st.caption(
    "Software-only decision support • congestion & incident detection • 15–60 min forecasting • "
    "adaptive advisories • infrastructure recommendations • explainable & confidence-aware"
)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    all_ts = sorted(traffic["timestamp"].unique())
    # default to latest
    ts_idx = st.slider("Snapshot time index", 0, len(all_ts) - 1, len(all_ts) - 1, help="Move through validation timeline")
    selected_ts = all_ts[ts_idx]
    st.write(f"**Selected:** {pd.Timestamp(selected_ts).strftime('%Y-%m-%d %H:%M')}")
    sev_filter = st.multiselect(
        "Severity filter (map & tables)",
        ["Severe", "High", "Moderate", "Normal"],
        default=["Severe", "High", "Moderate", "Normal"],
    )
    show_incidents = st.checkbox("Overlay known incidents (validation)", True)
    show_roadworks = st.checkbox("Overlay roadworks", True)
    st.markdown("---")
    st.markdown("**Model status**")
    if model_bundle:
        st.success(f"Loaded · {model_bundle.get('version', 'v1')}")
        if "metrics" in model_bundle:
            for h, m in model_bundle["metrics"].items():
                st.caption(f"+{h}m val MAE {m['val_mae']}")
    else:
        st.warning("Run `python train.py` to enable forecasts")

# Build snapshot
snap = traffic[traffic.timestamp == selected_ts].copy()
snap = snap.merge(
    net[
        [
            "segment_id",
            "free_flow_speed_kmh",
            "capacity_vph",
            "road_class",
            "src_lat",
            "src_lon",
            "dst_lat",
            "dst_lon",
            "importance",
            "structural_bottleneck",
            "lanes",
            "length_km",
        ]
    ],
    on="segment_id",
    how="left",
)
snap["severity"] = snap["congestion_index"].apply(severity)
snap["anomaly"] = anomaly_score(snap)
snap = snap[snap["severity"].isin(sev_filter)] if sev_filter else snap

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Network snapshot", pd.Timestamp(selected_ts).strftime("%d %b %H:%M"))
c2.metric("Severe segments", int((snap.congestion_index >= 0.65).sum()))
c3.metric("High+ segments", int((snap.congestion_index >= 0.40).sum()))
c4.metric("Avg speed", f"{snap.speed_kmh.mean():.1f} km/h")
c5.metric("Avg congestion", f"{snap.congestion_index.mean():.0%}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🗺️ Network Map",
        "🔮 Forecast & Advisory",
        "⚠️ Detection & Incidents",
        "🏗️ Infrastructure Planning",
        "📊 Explainability",
        "🧪 Robustness / Scenarios",
    ]
)

# ---- Tab 1: Network Map ----
with tab1:
    st.subheader("Current network state")
    fig = go.Figure()
    for sev, g in snap.groupby("severity"):
        for _, r in g.iterrows():
            fig.add_trace(
                go.Scattermap(
                    lat=[r.src_lat, r.dst_lat],
                    lon=[r.src_lon, r.dst_lon],
                    mode="lines",
                    line={"width": 4 if sev in ("Severe", "High") else 2.5, "color": severity_color(sev)},
                    hovertext=(
                        f"{r.segment_id} | {sev}<br>"
                        f"Speed {r.speed_kmh:.1f} km/h · CI {r.congestion_index:.0%}<br>"
                        f"Queue {r.queue_length_veh:.0f} · Occ {r.occupancy_pct:.0f}%"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                    name=sev,
                )
            )
    # Legend proxy
    for sev in ["Normal", "Moderate", "High", "Severe"]:
        fig.add_trace(
            go.Scattermap(
                lat=[None],
                lon=[None],
                mode="lines",
                line={"width": 4, "color": severity_color(sev)},
                name=sev,
            )
        )
    # Incidents overlay
    if show_incidents and not incidents.empty:
        active = incidents[
            (incidents.start_time <= selected_ts) & (incidents.end_time >= selected_ts)
        ]
        if not active.empty:
            inc_net = active.merge(net[["segment_id", "src_lat", "src_lon", "dst_lat", "dst_lon"]], on="segment_id", how="left")
            for _, r in inc_net.iterrows():
                if pd.notna(r.src_lat):
                    fig.add_trace(
                        go.Scattermap(
                            lat=[(r.src_lat + r.dst_lat) / 2],
                            lon=[(r.src_lon + r.dst_lon) / 2],
                            mode="markers",
                            marker={"size": 12, "color": "#7c3aed", "symbol": "triangle"},
                            hovertext=f"INCIDENT {r.incident_id}<br>{r.incident_type} sev={r.severity}",
                            hoverinfo="text",
                            name="Incident",
                            showlegend=False,
                        )
                    )
    if show_roadworks and not roadworks.empty:
        active_rw = roadworks[
            (roadworks.start_time <= selected_ts) & (roadworks.end_time >= selected_ts)
        ]
        if not active_rw.empty:
            rw_net = active_rw.merge(net[["segment_id", "src_lat", "src_lon", "dst_lat", "dst_lon"]], on="segment_id", how="left")
            for _, r in rw_net.iterrows():
                if pd.notna(r.src_lat):
                    fig.add_trace(
                        go.Scattermap(
                            lat=[(r.src_lat + r.dst_lat) / 2],
                            lon=[(r.src_lon + r.dst_lon) / 2],
                            mode="markers",
                            marker={"size": 11, "color": "#0ea5e9", "symbol": "square"},
                            hovertext=f"ROADWORK {r.work_id}<br>{r.work_type} closure={r.closure_fraction:.0%}",
                            hoverinfo="text",
                            name="Roadwork",
                            showlegend=False,
                        )
                    )

    center_lat = float(snap.src_lat.mean()) if len(snap) else 17.4
    center_lon = float(snap.src_lon.mean()) if len(snap) else 78.5
    fig.update_layout(
        map={"style": "open-street-map", "center": {"lat": center_lat, "lon": center_lon}, "zoom": 11},
        height=580,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        snap[
            [
                "segment_id",
                "road_class",
                "speed_kmh",
                "flow_vph",
                "occupancy_pct",
                "queue_length_veh",
                "congestion_index",
                "severity",
                "anomaly",
                "sensor_quality",
            ]
        ]
        .sort_values("congestion_index", ascending=False)
        .head(30),
        use_container_width=True,
        hide_index=True,
    )

# ---- Tab 2: Forecast & Advisory ----
with tab2:
    st.subheader("15–60 minute congestion forecast & operational advisory")
    seg_list = snap.sort_values("congestion_index", ascending=False)["segment_id"].tolist()
    if not seg_list:
        st.info("No segments match current filters.")
    else:
        seg = st.selectbox("Choose road segment", seg_list, key="fc_seg")
        r = snap[snap.segment_id == seg].iloc[0]
        hist = traffic[traffic.segment_id == seg].sort_values("timestamp").tail(36)

        if model_bundle:
            X = make_features(pd.DataFrame([r]), context=context)[model_bundle["features"]]
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            preds = {
                h: float(np.clip(model_bundle["models"][h].predict(X)[0], 0, 1)) for h in [15, 30, 45, 60]
            }
        else:
            preds = {h: float(r.congestion_index) for h in [15, 30, 45, 60]}
            st.warning("Model file missing. Run: python train.py")

        adv = generate_advisory(r, preds, net, neighbors, snap)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current CI", f"{r.congestion_index:.0%}", severity(r.congestion_index))
        m2.metric("30-min forecast", f"{preds[30]:.0%}", severity(preds[30]))
        m3.metric("Queue", f"{r.queue_length_veh:.0f} veh")
        m4.metric("Advisory confidence", f"{adv['confidence']:.0%}")

        chart = pd.DataFrame(
            {
                "Horizon": ["Now", "+15m", "+30m", "+45m", "+60m"],
                "Congestion": [r.congestion_index, preds[15], preds[30], preds[45], preds[60]],
            }
        )
        fig_fc = px.line(
            chart,
            x="Horizon",
            y="Congestion",
            markers=True,
            range_y=[0, 1],
            title=f"Forecast trajectory — {seg}",
        )
        fig_fc.add_hline(y=0.65, line_dash="dash", line_color="red", annotation_text="Severe")
        fig_fc.add_hline(y=0.40, line_dash="dash", line_color="orange", annotation_text="High")
        st.plotly_chart(fig_fc, use_container_width=True)

        # Historical sparkline
        if len(hist) > 2:
            fig_h = px.line(
                hist,
                x="timestamp",
                y="congestion_index",
                title=f"Recent history — {seg}",
                range_y=[0, 1],
            )
            st.plotly_chart(fig_h, use_container_width=True)

        st.markdown(f"### Advisory priority: **{adv['priority']}**")
        st.info(f"**Expected impact:** {adv['impact']}")
        for a in adv["actions"]:
            st.markdown(f"- {a}")
        with st.expander("Evidence & reasoning"):
            st.write(adv["reasoning"])
            st.caption(f"Incident signal: {adv['incident_label']} (score {adv['incident_score']}) — {adv['incident_evidence']}")
            if adv["free_neighbors"]:
                st.write("Lower-load neighbors considered:", ", ".join(adv["free_neighbors"]))
        st.caption("All advisories are simulated decision support only and do not control live infrastructure.")

# ---- Tab 3: Detection & Incidents ----
with tab3:
    st.subheader("Congestion, anomaly & incident detection")
    st.markdown(
        "Transparent multi-signal anomaly score + heuristic incident likelihood. "
        "Designed to control false alarms by requiring multiple stress indicators."
    )

    # Top anomalies
    top_anom = snap.nlargest(20, "anomaly")[
        ["segment_id", "speed_kmh", "flow_vph", "occupancy_pct", "queue_length_veh", "congestion_index", "anomaly", "sensor_quality"]
    ]
    st.write("**Top stress / anomaly ranking (latest snapshot)**")
    st.dataframe(top_anom, use_container_width=True, hide_index=True)

    # Incident likelihood table
    rows = []
    for _, row in snap.nlargest(40, "congestion_index").iterrows():
        lab, sc, ev = classify_incident_likelihood(row)
        rows.append(
            {
                "segment_id": row.segment_id,
                "CI": round(row.congestion_index, 3),
                "incident_label": lab,
                "likelihood": sc,
                "evidence": ev,
                "sensor_quality": row.sensor_quality,
            }
        )
    inc_df = pd.DataFrame(rows)
    st.write("**Incident likelihood (heuristic, evidence-based)**")
    st.dataframe(inc_df[inc_df.likelihood >= 0.25], use_container_width=True, hide_index=True)

    # Spillback
    spill = detect_spillback(snap, neighbors)
    st.write("**Spillback risk (congested segment + congested neighbors)**")
    if spill.empty:
        st.success("No strong spillback clusters detected in this snapshot.")
    else:
        st.dataframe(spill.head(15), use_container_width=True, hide_index=True)

    # Known validation incidents at this time
    if not incidents.empty:
        active = incidents[(incidents.start_time <= selected_ts) & (incidents.end_time >= selected_ts)]
        st.write(f"**Organizer-labelled incidents active at snapshot:** {len(active)}")
        if not active.empty:
            st.dataframe(active, use_container_width=True, hide_index=True)

    if not roadworks.empty:
        active_rw = roadworks[(roadworks.start_time <= selected_ts) & (roadworks.end_time >= selected_ts)]
        st.write(f"**Active roadworks:** {len(active_rw)}")
        if not active_rw.empty:
            st.dataframe(active_rw, use_container_width=True, hide_index=True)

# ---- Tab 4: Infrastructure Planning ----
with tab4:
    st.subheader("Data-driven infrastructure & network modification suggestions")
    st.markdown(
        "Recurring bottlenecks (elevated average congestion on validation window) are matched to "
        "`planning_candidates.csv`. Estimated before/after impact is a transparent heuristic based on "
        "capacity delta and current load — for decision support only."
    )
    sug = infrastructure_suggestions(snap, planning, traffic)
    if sug.empty:
        st.warning("No matching planning candidates for high-recurring segments, or planning file missing.")
    else:
        st.dataframe(sug, use_container_width=True, hide_index=True)
        st.markdown("**How to read**")
        st.markdown(
            "- `avg_ci` / `max_ci`: observed recurring load on the target segment  \n"
            "- `est_ci_reduction`: approximate expected drop in average congestion index  \n"
            "- `est_after_ci`: projected residual congestion after intervention  \n"
            "- `priority_score`: balances severity, impact and cost/feasibility  \n"
            "- All numbers are advisory estimates, not engineering guarantees."
        )

    # Structural bottlenecks from network
    if "structural_bottleneck" in net.columns:
        bott = net[net["structural_bottleneck"] == 1][["segment_id", "road_class", "lanes", "capacity_vph", "importance"]]
        if not bott.empty:
            st.write("**Network-declared structural bottlenecks**")
            st.dataframe(bott, use_container_width=True, hide_index=True)

# ---- Tab 5: Explainability ----
with tab5:
    st.subheader("Explainability & confidence handling")
    st.markdown(
        """
**Feature set used by the forecaster** (no target leakage):
"""
    )
    st.code(", ".join(FEATURES), language=None)
    st.markdown(
        """
**Congestion severity thresholds** (transparent & fixed):  
- Normal < 0.20 · Moderate ≥ 0.20 · High ≥ 0.40 · Severe ≥ 0.65  

**Anomaly score** (weighted, observable only):  
`0.40·CI + 0.20·occupancy + 0.20·queue + 0.10·delay + 0.10·(1−sensor_quality)`

**Incident likelihood** combines high CI + long queues + high occupancy/low speed + delay; 
score is reduced when sensor quality is low.

**Forecast confidence** is derived from sensor_quality and observation support.  
Low-confidence forecasts are still shown but flagged so operators can discount them.

**Limitations**  
- Models are trained on organizer data; abrupt regime shifts (major events, extreme weather) 
  may degrade accuracy.  
- Diversion recommendations do not solve a full network assignment; they use local neighbor load.  
- Infrastructure impact estimates are first-order heuristics, not microsimulation.
"""
    )
    if model_bundle and "metrics" in model_bundle:
        st.write("**Held-out validation metrics (from train.py)**")
        st.json(model_bundle["metrics"])

# ---- Tab 6: Robustness ----
with tab6:
    st.subheader("Robustness to unseen patterns & noisy data")
    st.markdown(
        """
The system is designed to degrade gracefully:

1. **Missing / low-quality sensors** — `sensor_quality` is an explicit feature; anomaly and 
   confidence scores down-weight unreliable segments.  
2. **Demand / weather shifts** — time-of-day, peak flag, rain and event_level from context are 
   included so the model can adapt within the observed distribution.  
3. **Incidents & roadworks** — detection layer surfaces abnormal stress even when the 
   regression forecast has not yet fully reacted.  
4. **Spillback** — neighbor-aware checks catch congestion that is no longer local.

**What-if style inspection**  
Select a segment and artificially stress key observables to see how advisory priority changes 
(purely illustrative, does not retrain).
"""
    )
    if seg_list:
        wseg = st.selectbox("Segment for stress test", seg_list, key="whatif")
        base_row = snap[snap.segment_id == wseg].iloc[0].copy()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            w_ci = st.slider("Injected congestion index", 0.0, 1.0, float(base_row.congestion_index), 0.05)
        with col_b:
            w_q = st.slider("Injected queue (veh)", 0, 80, int(base_row.queue_length_veh), 5)
        with col_c:
            w_occ = st.slider("Injected occupancy %", 0, 100, int(base_row.occupancy_pct), 5)
        test = base_row.copy()
        test["congestion_index"] = w_ci
        test["queue_length_veh"] = w_q
        test["occupancy_pct"] = w_occ
        if model_bundle:
            Xw = make_features(pd.DataFrame([test]), context=context)[model_bundle["features"]].fillna(0)
            preds_w = {h: float(np.clip(model_bundle["models"][h].predict(Xw)[0], 0, 1)) for h in [15, 30, 45, 60]}
        else:
            preds_w = {h: w_ci for h in [15, 30, 45, 60]}
        adv_w = generate_advisory(test, preds_w, net, neighbors, snap)
        st.markdown(f"**Resulting priority:** `{adv_w['priority']}` · confidence {adv_w['confidence']:.0%}")
        st.write(adv_w["reasoning"])
        for a in adv_w["actions"]:
            st.markdown(f"- {a}")

st.markdown("---")
st.caption(
    "NeuraX · software-only AI decision support · no live signal control, cameras, GPS devices or municipal actuation. "
    "All diversions and infrastructure suggestions are advisory / simulated."
)
