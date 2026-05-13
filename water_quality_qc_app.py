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
    detect_timestamp_column,
    align_to_grid,
)

# ---------------------------------------------------------------------------
# Robust CSV reading (handles Excel exports: UTF-16, Latin-1, tabs, etc.)
# ---------------------------------------------------------------------------

def read_csv_resilient(uploaded_file, label: str = "file") -> tuple[pd.DataFrame, str]:
    """Read a CSV upload trying common encodings and delimiters.

    Excel-exported CSVs are often UTF-16 (with a BOM) or Windows-1252,
    not UTF-8. Tab-separated exports also happen. This tries each
    combination and returns the first that parses to >= 2 columns.

    Returns (dataframe, note) where `note` describes which encoding worked.
    """
    encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]
    separators = [",", "\t", ";"]

    last_err = None
    for enc in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    note = f"encoding={enc}"
                    if sep != ",":
                        note += f", sep={'TAB' if sep == chr(9) else repr(sep)}"
                    return df, note
            except (UnicodeDecodeError, UnicodeError) as e:
                last_err = e
                break  # encoding wrong → no point trying more separators
            except Exception as e:
                last_err = e
                continue
    raise ValueError(
        f"Could not read {label}. Tried encodings {encodings} and "
        f"separators {separators}. Last error: {last_err}"
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
if "alignment_log" not in st.session_state:
    st.session_state.alignment_log = []

# ---------------------------------------------------------------------------
# Sidebar: data sources (WQ + optional rainfall + optional stage)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Data sources")

    source = st.radio(
        "Choose data source",
        ["Upload CSV files", "Try demo data"],
        label_visibility="collapsed",
    )

    if source == "Try demo data":
        if st.button("Generate demo data", use_container_width=True):
            st.session_state.data = generate_demo_data()
            st.session_state.alignment_log = [
                "Demo data: rainfall + stage already on the WQ grid (no alignment needed)."
            ]
            st.success(f"Generated {len(st.session_state.data):,} rows of demo data")
    else:
        st.markdown("**Water quality CSV** *(required)*")
        wq_file = st.file_uploader(
            "Timestamp + parameter columns",
            type=["csv"], key="wq_upload",
            label_visibility="collapsed",
        )

        st.markdown("**Rainfall CSV** *(optional, separate file)*")
        rain_file = st.file_uploader(
            "Timestamp + rainfall column",
            type=["csv"], key="rain_upload",
            label_visibility="collapsed",
        )

        st.markdown("**Stage CSV** *(optional, separate file)*")
        stage_file = st.file_uploader(
            "Timestamp + stage column",
            type=["csv"], key="stage_upload",
            label_visibility="collapsed",
        )

        tol_min = st.number_input(
            "Timestamp alignment tolerance (minutes)",
            min_value=0.5, max_value=120.0, value=10.0, step=0.5,
            help=(
                "When matching rainfall/stage to the WQ grid, samples outside "
                "this window become NaN. Use ~half your WQ sampling interval "
                "as a starting point (e.g. 7.5 min for 15-min WQ data)."
            ),
        )

        if st.button("📥 Load & align files", use_container_width=True,
                     disabled=wq_file is None):
            log = []
            try:
                wq_df, wq_note = read_csv_resilient(wq_file, "water quality file")
            except Exception as e:
                st.error(f"Couldn't read the WQ file: {e}")
                st.stop()
            wq_ts = detect_timestamp_column(wq_df)
            if wq_ts is None:
                st.error("Couldn't auto-detect a timestamp column in the WQ file.")
                st.stop()
            log.append(
                f"WQ: {len(wq_df):,} rows, {wq_note}. "
                f"Timestamp column: '{wq_ts}'."
            )

            merged = wq_df.copy()

            # ---- Rainfall alignment ----
            if rain_file is not None:
                try:
                    rain_df, rain_note = read_csv_resilient(rain_file, "rainfall file")
                except Exception as e:
                    log.append(f"⚠️ Rainfall file: couldn't read — {e}. Skipped.")
                    rain_df = None
                if rain_df is not None:
                    rain_ts = detect_timestamp_column(rain_df)
                    if rain_ts is None:
                        log.append(
                            f"⚠️ Rainfall file ({rain_note}): couldn't detect "
                            f"timestamp column — skipped."
                        )
                    else:
                        rain_value_cols = [c for c in rain_df.columns if c != rain_ts]
                        merged, diag = align_to_grid(
                            merged, wq_ts, rain_df, rain_ts,
                            rain_value_cols, tolerance_minutes=tol_min,
                        )
                        log.append(
                            f"Rainfall: {len(rain_df):,} rows, {rain_note}. "
                            f"Timestamp column: '{rain_ts}'."
                        )
                        for vc in rain_value_cols:
                            log.append(
                                f"  └ '{vc}': {diag[f'{vc}_matched']:,} / "
                                f"{diag['target_rows']:,} matched "
                                f"({diag[f'{vc}_match_pct']}%) within {tol_min} min."
                            )

            # ---- Stage alignment ----
            if stage_file is not None:
                try:
                    stage_df, stage_note = read_csv_resilient(stage_file, "stage file")
                except Exception as e:
                    log.append(f"⚠️ Stage file: couldn't read — {e}. Skipped.")
                    stage_df = None
                if stage_df is not None:
                    stage_ts = detect_timestamp_column(stage_df)
                    if stage_ts is None:
                        log.append(
                            f"⚠️ Stage file ({stage_note}): couldn't detect "
                            f"timestamp column — skipped."
                        )
                    else:
                        stage_value_cols = [c for c in stage_df.columns if c != stage_ts]
                        merged, diag = align_to_grid(
                            merged, wq_ts, stage_df, stage_ts,
                            stage_value_cols, tolerance_minutes=tol_min,
                        )
                        log.append(
                            f"Stage: {len(stage_df):,} rows, {stage_note}. "
                            f"Timestamp column: '{stage_ts}'."
                        )
                        for vc in stage_value_cols:
                            log.append(
                                f"  └ '{vc}': {diag[f'{vc}_matched']:,} / "
                                f"{diag['target_rows']:,} matched "
                                f"({diag[f'{vc}_match_pct']}%) within {tol_min} min."
                            )

            st.session_state.data = merged
            st.session_state.alignment_log = log
            st.success(f"Loaded and aligned: {len(merged):,} rows × {len(merged.columns)} columns.")

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

# ----- Alignment log ------------------------------------------------------

if st.session_state.alignment_log:
    with st.expander("🔗 File alignment log", expanded=True):
        for line in st.session_state.alignment_log:
            st.markdown(f"- {line}")
        st.caption(
            "Tip: if match % is lower than expected, your timestamps may be in "
            "different timezones, or the tolerance may be too tight. "
            "Unmatched rows become NaN, which the QC engine handles safely."
        )

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

# ----- Covariate mapping (rainfall + stage) -------------------------------

st.subheader("2b. Covariates (rainfall + stage)")
st.caption(
    "Optional but powerful. Rainfall and stage are used to (1) suppress "
    "false-positive flags during real hydrologic events and (2) inform "
    "correction estimates via regression."
)

cv1, cv2 = st.columns(2)
with cv1:
    rainfall_col = st.selectbox(
        "Rainfall column (e.g. inches per timestep)",
        options=["— none —"] + [c for c in data.columns if c not in (ts_col,)],
        index=(
            (["— none —"] + [c for c in data.columns if c not in (ts_col,)]).index("rainfall")
            if "rainfall" in data.columns else 0
        ),
    )
    rain_window_hr = st.number_input(
        "Rolling window for rain events (hr)",
        min_value=0.25, max_value=24.0, value=1.0, step=0.25,
    )
    rain_event_threshold = st.number_input(
        "Rain total over window to call it an 'event' (same units as rainfall col)",
        min_value=0.0, value=0.05, step=0.01, format="%.3f",
    )
with cv2:
    stage_col = st.selectbox(
        "Stage column (PT or radar, ft/m)",
        options=["— none —"] + [c for c in data.columns if c not in (ts_col,)],
        index=(
            (["— none —"] + [c for c in data.columns if c not in (ts_col,)]).index("stage")
            if "stage" in data.columns else 0
        ),
    )
    stage_high_quantile = st.slider(
        "Stage quantile considered 'high stage'",
        min_value=0.50, max_value=0.99, value=0.90, step=0.01,
    )

# Convert "none" sentinel to None
rainfall_col = None if rainfall_col == "— none —" else rainfall_col
stage_col = None if stage_col == "— none —" else stage_col

# ----- Parameter configs (collapsible) ------------------------------------

st.subheader("3. Configure detection thresholds")
st.caption("Defaults are sensible for typical aquatic sensors — tweak per parameter if needed.")

import math


def _safe_float(value: float, fallback: float) -> float:
    """Streamlit's number_input cannot render NaN or infinity.
    Clamp those to a finite fallback before passing to the widget."""
    if value is None or math.isnan(value) or math.isinf(value):
        return fallback
    return float(value)


custom_configs: dict[str, ParameterConfig] = {}
for p in param_cols:
    default = PARAMETER_CONFIGS.get(p, ParameterConfig(name=p))
    with st.expander(f"⚙️ {p} ({default.units or 'no units'})"):
        c1, c2, c3 = st.columns(3)
        with c1:
            rmin = st.number_input(
                "Min range",
                value=_safe_float(default.range_min, -1e6),
                key=f"{p}_min",
            )
            rmax = st.number_input(
                "Max range",
                value=_safe_float(default.range_max, 1e6),
                key=f"{p}_max",
            )
        with c2:
            spike = st.number_input(
                "Spike z-threshold",
                value=_safe_float(default.spike_threshold, 4.0),
                key=f"{p}_spike",
            )
            pers = st.number_input(
                "Persistence window",
                value=int(default.persistence_window),
                step=1,
                key=f"{p}_pers",
            )
        with c3:
            rate = st.number_input(
                "Max rate/hr",
                value=_safe_float(default.max_rate_change, 1e6),
                key=f"{p}_rate",
            )
            arima = st.checkbox("Use ARIMA", value=default.use_arima, key=f"{p}_arima")

        st.markdown("**Event-aware behavior** (uses rainfall / stage)")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            sup_spike_rain = st.checkbox(
                "Suppress spike flags during rain",
                value=default.suppress_spike_in_rain,
                key=f"{p}_sup_spike",
                disabled=rainfall_col is None,
            )
        with e2:
            sup_rate_rain = st.checkbox(
                "Suppress rate flags during rain",
                value=default.suppress_rate_in_rain,
                key=f"{p}_sup_rate",
                disabled=rainfall_col is None,
            )
        with e3:
            sup_min_stage = st.checkbox(
                "Suppress range-min flags during high stage",
                value=default.suppress_range_min_in_high_stage,
                key=f"{p}_sup_min",
                disabled=stage_col is None,
            )
        with e4:
            use_cov = st.checkbox(
                "Use covariates for correction",
                value=default.use_covariates_for_correction,
                key=f"{p}_use_cov",
                disabled=(rainfall_col is None and stage_col is None),
            )

        custom_configs[p] = ParameterConfig(
            name=p, units=default.units,
            range_min=rmin, range_max=rmax,
            spike_threshold=spike, persistence_window=pers,
            max_rate_change=rate, use_arima=arima,
            suppress_spike_in_rain=sup_spike_rain,
            suppress_rate_in_rain=sup_rate_rain,
            suppress_range_min_in_high_stage=sup_min_stage,
            use_covariates_for_correction=use_cov,
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
            rainfall_col=rainfall_col,
            stage_col=stage_col,
            rain_window_hr=rain_window_hr,
            rain_event_threshold=rain_event_threshold,
            stage_high_quantile=stage_high_quantile,
        )
        qc.run_all_sequential()
        qc.correct_all()
        st.session_state.qc = qc
    st.success("✅ QC + correction complete")

    # Event summary
    if rainfall_col or stage_col:
        ev_msg = []
        if rainfall_col:
            ev_msg.append(
                f"{int(qc.rain_event.sum())} samples in rain events "
                f"({100*qc.rain_event.mean():.1f}%)"
            )
        if stage_col:
            ev_msg.append(
                f"{int(qc.high_stage.sum())} samples in high stage "
                f"({100*qc.high_stage.mean():.1f}%)"
            )
        st.info("**Hydrologic events detected:** " + "  •  ".join(ev_msg))

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
    st.markdown("#### Plots — flagged points highlighted, events shaded")
    for p in qc.parameters:
        if p not in qc.flags:
            continue
        flags = qc.flags[p]
        series = qc.data[p]
        ts = qc.data[ts_col]

        fig, ax = plt.subplots(figsize=(12, 3.2))

        # Shade rain-event and high-stage periods
        if rainfall_col:
            ax.fill_between(
                ts, series.min(), series.max(),
                where=qc.rain_event.values, alpha=0.10, color="steelblue",
                step="mid", label="rain event", linewidth=0,
            )
        if stage_col:
            ax.fill_between(
                ts, series.min(), series.max(),
                where=qc.high_stage.values, alpha=0.10, color="goldenrod",
                step="mid", label="high stage", linewidth=0,
            )

        # Raw and corrected
        ax.plot(ts, series, lw=0.7, color="#1f77b4", label="raw")
        if p in qc.corrected:
            ax.plot(
                ts, qc.corrected[p],
                lw=0.7, color="#2ca02c", alpha=0.7, label="corrected",
            )

        # Flagged points
        flagged_mask = flags["any"]
        if flagged_mask.any():
            ax.scatter(
                ts[flagged_mask], series[flagged_mask],
                color="red", s=14, label="flagged", zorder=3,
            )

        # Suppressed-by-events points (rendered as hollow circles)
        if "suppressed" in flags.columns and flags["suppressed"].any():
            sup_mask = flags["suppressed"]
            ax.scatter(
                ts[sup_mask], series[sup_mask],
                facecolors="none", edgecolors="orange", s=18,
                label="suppressed (real event)", zorder=2,
            )

        ax.set_title(f"{p} ({qc.configs[p].units})")
        ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ----- Downloads ------------------------------------------------------

    st.subheader("6. Download results")

    # Build everything in-memory for download
    csv_buf = io.BytesIO()
    out_df = qc.data.copy()
    for p, f in qc.flags.items():
        for col in ["range", "spike", "persistence", "rate_of_change",
                    "arima", "any", "suppressed"]:
            if col in f.columns:
                out_df[f"{p}_flag_{col}"] = f[col].values
        if p in qc.corrected:
            out_df[f"{p}_corrected"] = qc.corrected[p].values
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
