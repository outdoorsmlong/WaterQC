"""
Water Quality QC - Streamlit UI
================================
A beginner-friendly web interface for the water_quality_qc_v2 framework.

Run with:
    streamlit run water_quality_qc_app.py

The UI handles file upload, parameter config, and visualization.
The Python framework (water_quality_qc_v2.py) runs in the background.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from water_quality_qc_v2 import (
    WaterQualityQCv2,
    ParameterConfig,
    PARAMETER_CONFIGS,
    generate_demo_data,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Water Quality QC",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💧 Water Quality QC Tool")
st.caption(
    "Upload sensor data, configure thresholds, and flag anomalies. "
    "The Python QC engine runs in the background — no coding required."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None
if "qc" not in st.session_state:
    st.session_state.qc = None

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Data source")

    source = st.radio(
        "Choose data source",
        ["Upload CSV", "Try demo data"],
        label_visibility="collapsed",
    )

    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "CSV with a timestamp column + parameter columns",
            type=["csv"],
        )
        if uploaded is not None:
            st.session_state.data = pd.read_csv(uploaded)
            st.success(f"Loaded {len(st.session_state.data):,} rows")
    else:
        if st.button("Generate demo data", use_container_width=True):
            st.session_state.data = generate_demo_data()
            st.success(f"Generated {len(st.session_state.data):,} rows of demo data")

# ---------------------------------------------------------------------------
# Main area: only show once data is loaded
# ---------------------------------------------------------------------------

if st.session_state.data is None:
    st.info(
        "👈 Load data from the sidebar to get started. "
        "Click **Generate demo data** if you just want to try the tool."
    )
    st.stop()

data = st.session_state.data

# ----- Preview ------------------------------------------------------------

with st.expander("📋 Data preview", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")

# ----- Column mapping -----------------------------------------------------

st.subheader("2. Map your columns")

col1, col2 = st.columns([1, 2])
with col1:
    ts_col = st.selectbox(
        "Timestamp column",
        options=data.columns.tolist(),
        index=0,
    )
with col2:
    param_cols = st.multiselect(
        "Parameter columns to QC",
        options=[c for c in data.columns if c != ts_col],
        default=[c for c in data.columns if c in PARAMETER_CONFIGS],
    )

# ----- Parameter configs (collapsible) ------------------------------------

st.subheader("3. Configure detection thresholds")
st.caption("Defaults are sensible for typical aquatic sensors — tweak per parameter if needed.")

custom_configs: dict[str, ParameterConfig] = {}
for p in param_cols:
    default = PARAMETER_CONFIGS.get(p, ParameterConfig(name=p))
    with st.expander(f"⚙️ {p} ({default.units or 'no units'})"):
        c1, c2, c3 = st.columns(3)
        with c1:
            rmin = st.number_input(f"Min range", value=float(default.range_min), key=f"{p}_min")
            rmax = st.number_input(f"Max range", value=float(default.range_max), key=f"{p}_max")
        with c2:
            spike = st.number_input(f"Spike z-threshold", value=float(default.spike_threshold), key=f"{p}_spike")
            pers = st.number_input(f"Persistence window", value=int(default.persistence_window), step=1, key=f"{p}_pers")
        with c3:
            rate = st.number_input(f"Max rate/hr", value=float(default.max_rate_change), key=f"{p}_rate")
            arima = st.checkbox(f"Use ARIMA", value=default.use_arima, key=f"{p}_arima")
        custom_configs[p] = ParameterConfig(
            name=p, units=default.units,
            range_min=rmin, range_max=rmax,
            spike_threshold=spike, persistence_window=pers,
            max_rate_change=rate, use_arima=arima,
        )

# ----- Run button ---------------------------------------------------------

st.subheader("4. Run QC")

if st.button("🚀 Run QC checks", type="primary", use_container_width=True):
    if not param_cols:
        st.error("Pick at least one parameter to QC.")
        st.stop()

    with st.spinner("Running QC engine in the background..."):
        qc = WaterQualityQCv2(
            data=data,
            timestamp_col=ts_col,
            parameters=param_cols,
            configs=custom_configs,
        )
        qc.run_all_sequential()
        st.session_state.qc = qc
    st.success("✅ QC complete")

# ----- Results -------------------------------------------------------------

if st.session_state.qc is not None:
    qc = st.session_state.qc

    st.subheader("5. Results")

    # Summary table
    summary = qc.summary()
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Headline metrics
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        col.metric(
            label=row["parameter"],
            value=f"{row['any']:,} flagged",
            delta=f"{row['pct_flagged']}% of {row['n_records']:,}",
            delta_color="off",
        )

    # Per-parameter plots
    st.markdown("#### Plots — flagged points highlighted")
    for p in qc.parameters:
        if p not in qc.flags:
            continue
        flags = qc.flags[p]
        series = qc.data[p]
        ts = qc.data[ts_col]

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ts, series, lw=0.7, color="#1f77b4", label=p)
        flagged_mask = flags["any"]
        ax.scatter(
            ts[flagged_mask], series[flagged_mask],
            color="red", s=12, label="flagged", zorder=3,
        )
        ax.set_title(f"{p} ({qc.configs[p].units})")
        ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ----- Downloads ------------------------------------------------------

    st.subheader("6. Download results")

    # Build everything in-memory for download
    csv_buf = io.BytesIO()
    out_df = qc.data.copy()
    for p, f in qc.flags.items():
        for col in f.columns:
            out_df[f"{p}_flag_{col}"] = f[col].values
    out_df.to_csv(csv_buf, index=False)

    # Build summary text
    summary_text_lines = [
        "WATER QUALITY QC SUMMARY REPORT",
        "=" * 40,
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        f"Total records: {len(qc.data):,}",
        "",
    ]
    for _, r in summary.iterrows():
        summary_text_lines += [
            f"{r['parameter'].upper()} ({r['units']})",
            f"  Total anomalies: {r['any']} ({r['pct_flagged']}%)",
            f"    - range:          {r['range']}",
            f"    - spike:          {r['spike']}",
            f"    - persistence:    {r['persistence']}",
            f"    - rate_of_change: {r['rate_of_change']}",
            f"    - arima:          {r['arima']}",
            "",
        ]
    summary_text = "\n".join(summary_text_lines)

    # Zip everything
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("flagged_data.csv", csv_buf.getvalue())
        z.writestr("summary_report.txt", summary_text)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇️ Flagged CSV", data=csv_buf.getvalue(),
        file_name="flagged_data.csv", mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "⬇️ Summary report", data=summary_text,
        file_name="summary_report.txt", mime="text/plain",
        use_container_width=True,
    )
    c3.download_button(
        "⬇️ All results (zip)", data=zip_buf.getvalue(),
        file_name="qc_results.zip", mime="application/zip",
        use_container_width=True,
    )
