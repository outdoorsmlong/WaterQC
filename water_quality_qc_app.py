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
from dataclasses import asdict
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

from preset_loader import (
    list_presets, configs_from_preset, auto_map_columns,
    preset_to_session_state,
)

# ML backend selection: prefer LSTM (TensorFlow) if available, otherwise
# fall back to sklearn-based learned models. Both expose the same API, so
# the rest of the app uses a generic name (`ParameterModel`) and doesn't
# care which one is active.
_ML_BACKEND = None
_ML_BACKEND_ERROR = ""

# Try sklearn (learned_models) first — it's the lighter, more reliable path
try:
    from learned_models import (
        LearnedConfig as ParameterConfig_ML,
        ParameterLearned as ParameterModel,
        derive_labels, compute_metrics,
        DEFAULT_LABEL_TOLERANCES,
    )
    _ML_BACKEND = "sklearn"
except ImportError as e:
    _ML_BACKEND_ERROR = f"scikit-learn missing: {e}"

# If the user explicitly has TF + lstm_models, they can override by setting
# this env var, but we don't try to load TF eagerly because that's slow.
import os as _os
if _os.environ.get("WQC_USE_LSTM", "").lower() in ("1", "true", "yes"):
    try:
        from lstm_models import (
            LSTMConfig as ParameterConfig_ML,
            ParameterLSTM as ParameterModel,
            derive_labels, compute_metrics,
            DEFAULT_LABEL_TOLERANCES,
        )
        import tensorflow as _tf_check  # noqa: F401
        _ML_BACKEND = "tensorflow_lstm"
    except ImportError as e:
        # Fall through — sklearn import above (if successful) stays active
        pass

_ML_AVAILABLE = _ML_BACKEND is not None

# Backward-compat aliases so the rest of the file doesn't need to change
_LSTM_AVAILABLE = _ML_AVAILABLE
_LSTM_IMPORT_ERROR = _ML_BACKEND_ERROR

# ---------------------------------------------------------------------------
# Robust CSV reading (handles Excel exports: UTF-16, Latin-1, tabs, etc.)
# ---------------------------------------------------------------------------

def read_csv_resilient(uploaded_file, label: str = "file") -> tuple[pd.DataFrame, str]:
    """Read a CSV or Excel upload, handling common real-world quirks.

    Quirks handled:
      - Multiple encodings (UTF-8, UTF-8-BOM, UTF-16, CP1252, Latin-1)
      - Multiple delimiters (comma, tab, semicolon)
      - Header row not at line 0 (AQUARIUS exports, Campbell exports)
      - Excel files (.xlsx) — read directly

    Returns (dataframe, note) describing what was detected so the user can see
    in the alignment log.
    """
    from water_quality_qc_v2 import detect_csv_header_row

    # Detect Excel by extension or content
    filename = getattr(uploaded_file, "name", "").lower()
    is_excel = filename.endswith((".xlsx", ".xls", ".xlsm"))

    if is_excel:
        # Excel: try the common AQUARIUS side-by-side layout first
        uploaded_file.seek(0)
        # Read all sheets, use the first
        xl = pd.ExcelFile(uploaded_file)
        sheet = xl.sheet_names[0]
        # AQUARIUS side-by-side: header is on row 2 (0-indexed)
        # Try row 0, 1, 2 and pick the one yielding the most named cols
        best_df, best_hdr = None, 0
        best_score = -1
        for hdr in (0, 1, 2):
            try:
                df_try = pd.read_excel(xl, sheet_name=sheet, header=hdr)
                # Score: count columns whose name is a non-empty string and not "Unnamed: N"
                score = sum(
                    1 for c in df_try.columns
                    if isinstance(c, str) and c and not c.startswith("Unnamed:")
                )
                if score > best_score:
                    best_score = score
                    best_df = df_try
                    best_hdr = hdr
            except Exception:
                continue
        if best_df is None or best_score < 2:
            raise ValueError(f"Could not parse Excel {label}: no readable header found.")
        note = f"format=Excel, sheet={sheet!r}, header_row={best_hdr}"
        return best_df, note

    # CSV path
    encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]
    separators = [",", "\t", ";"]

    # Auto-detect the header row (skips comments, UUID rows, units rows)
    try:
        hdr_row = detect_csv_header_row(uploaded_file)
    except Exception:
        hdr_row = 0

    last_err = None
    for enc in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    encoding=enc, sep=sep,
                    skiprows=hdr_row if hdr_row > 0 else None,
                )
                if df.shape[1] >= 2:
                    note = f"encoding={enc}"
                    if sep != ",":
                        note += f", sep={'TAB' if sep == chr(9) else repr(sep)}"
                    if hdr_row > 0:
                        note += f", header_row={hdr_row}"
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
            st.session_state._resample_decided = False
            st.session_state._resample_diag = None
            st.session_state.alignment_log = [
                "Demo data: rainfall + stage already on the WQ grid (no alignment needed)."
            ]
            st.success(f"Generated {len(st.session_state.data):,} rows of demo data")
    else:
        st.markdown("**Water quality CSV** *(required)*")
        wq_file = st.file_uploader(
            "Timestamp + parameter columns",
            type=["csv", "xlsx", "xls"], key="wq_upload",
            label_visibility="collapsed",
        )

        st.markdown("**Rainfall CSV** *(optional, separate file)*")
        rain_file = st.file_uploader(
            "Timestamp + rainfall column",
            type=["csv", "xlsx", "xls"], key="rain_upload",
            label_visibility="collapsed",
        )

        st.markdown("**Stage CSV** *(optional, separate file)*")
        stage_file = st.file_uploader(
            "Timestamp + stage column",
            type=["csv", "xlsx", "xls"], key="stage_upload",
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
            st.session_state._resample_decided = False
            st.session_state._resample_diag = None
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

# ----- Cadence detection + optional auto-resample -------------------------

from water_quality_qc_v2 import (
    detect_column_cadences, needs_resampling, resample_to_grid,
)

# Detect timestamp column (first column that parses as datetimes)
_probe_ts_col = None
for _c in data.columns:
    try:
        _parsed = pd.to_datetime(data[_c].head(50), errors="raise")
        if _parsed.notna().all():
            _probe_ts_col = _c
            break
    except Exception:
        continue

if _probe_ts_col is not None:
    _cadence_report = detect_column_cadences(data, _probe_ts_col)
    _mixed = needs_resampling(_cadence_report)
    if _mixed and not st.session_state.get("_resample_decided"):
        st.warning(
            "⚠️ **Mixed sampling cadences detected.** Some columns are at "
            "different intervals (likely a Campbell-logger-style export). "
            "If left unresampled, covariates on a different cadence than the "
            "WQ sensor will appear mostly NaN and won't contribute to QC."
        )
        with st.expander("Cadence details", expanded=True):
            st.dataframe(_cadence_report, use_container_width=True, hide_index=True)

        # Default target = median of cadences (excluding outliers)
        _valid_cads = _cadence_report.dropna(subset=["mode_interval_min"])
        _default_target = (
            float(_valid_cads["mode_interval_min"].median())
            if len(_valid_cads) else 15.0
        )

        rcol1, rcol2, rcol3 = st.columns([2, 2, 1])
        with rcol1:
            _target_interval = st.number_input(
                "Resample target interval (minutes)",
                min_value=1.0, max_value=240.0,
                value=_default_target, step=1.0,
                help="All columns will be regridded to this cadence. Rainfall is summed; everything else uses nearest-neighbor.",
            )
        with rcol2:
            # Guess rainfall column by name hint
            _rain_guess = [
                c for c in data.columns
                if any(h in c.lower() for h in ("precip", "rain", "rg"))
            ]
            _rain_cols_sel = st.multiselect(
                "Rainfall column(s) (will be SUMMED per bin)",
                options=[c for c in data.columns if c != _probe_ts_col],
                default=_rain_guess,
            )
        with rcol3:
            st.write("")  # spacer for alignment
            st.write("")
            if st.button("🔄 Auto-resample", type="primary", use_container_width=True):
                with st.spinner(f"Resampling to {_target_interval}-min grid..."):
                    new_data, diag = resample_to_grid(
                        data, _probe_ts_col,
                        target_interval_minutes=_target_interval,
                        rainfall_cols=_rain_cols_sel,
                    )
                st.session_state.data = new_data
                st.session_state._resample_decided = True
                st.session_state._resample_diag = {
                    "target_interval": _target_interval,
                    "n_rows_before": len(data),
                    "n_rows_after": len(new_data),
                    "rainfall_cols": _rain_cols_sel,
                }
                st.rerun()
            if st.button("Keep as-is", use_container_width=True):
                st.session_state._resample_decided = True
                st.session_state._resample_diag = {
                    "kept_mixed": True,
                    "n_rows": len(data),
                }
                st.rerun()
        st.stop()  # Don't proceed until user decides

    elif st.session_state.get("_resample_diag"):
        diag = st.session_state["_resample_diag"]
        if diag.get("kept_mixed"):
            st.info(
                f"📊 **Mixed-cadence file kept as-is** ({diag['n_rows']:,} rows). "
                "Some covariates may be NaN at WQ timestamps."
            )
        else:
            st.success(
                f"✅ **Resampled** from {diag['n_rows_before']:,} rows to "
                f"{diag['n_rows_after']:,} rows on a {diag['target_interval']}-min grid. "
                f"Rainfall cols summed: {diag.get('rainfall_cols') or 'none'}."
            )

# ===========================================================================
# Tabs: Rules-based QC  |  LSTM Train  |  LSTM Detect & Validate
# ===========================================================================

tab_rules, tab_train, tab_lstm = st.tabs([
    "🔧 Rules-based QC",
    "🧠 Train Model",
    "🔬 Detect & Validate",
])

with tab_rules:

    # ----- Station preset selector --------------------------------------------

    st.subheader("1b. Station preset (optional)")
    st.caption(
        "Load saved settings for a specific monitoring station. Presets pre-fill "
        "thresholds, column mappings, and event-aware behavior. You can still "
        "override any value below."
    )

    # Resolve presets folder relative to THIS file, not the current working
    # directory — so it works regardless of where Streamlit was launched from.
    _APP_DIR = Path(__file__).resolve().parent
    _PRESETS_DIR = _APP_DIR / "presets"

    presets = list_presets(_PRESETS_DIR)

    if not presets:
        st.warning(
            f"⚠️ **No presets found.** Expected JSON files in:  \n"
            f"`{_PRESETS_DIR}`  \n\n"
            "Common causes:\n"
            "- The `presets/` folder didn't get pushed to your repo or copied "
            "to the deployment.\n"
            "- The folder is there but the JSON files were not unzipped along "
            "with the Python files.\n\n"
            "**Fix:** make sure the `presets/` folder sits next to "
            "`water_quality_qc_app.py` and contains files like `KINA.json`, "
            "`SMIB.json`, `_generic_freshwater.json`."
        )
        # Diagnostic: list whatever IS at _APP_DIR so the user can self-debug
        try:
            siblings = sorted(p.name for p in _APP_DIR.iterdir())
            with st.expander("What's in the app folder?", expanded=False):
                st.code("\n".join(siblings) or "(empty)")
        except Exception as e:
            st.caption(f"(Couldn't list app folder: {e})")

    preset_labels = ["— No preset (use defaults) —"] + [
        f"{p['station_name']}"
        + (f"  ({p['n_parameters']} params)" if p['n_parameters'] else "")
        for p in presets
    ]
    preset_choice = st.selectbox(
        "Select a station preset",
        options=range(len(preset_labels)),
        format_func=lambda i: preset_labels[i],
        index=0,
        label_visibility="collapsed",
    )

    active_preset = None
    if preset_choice > 0:
        active_preset = presets[preset_choice - 1]
        if active_preset["_data"]:
            with st.expander(
                f"ℹ️ About: {active_preset['station_name']}", expanded=False
            ):
                if active_preset["description"]:
                    st.write(active_preset["description"])
                st.caption(
                    f"Source: `{active_preset['filename']}`  •  "
                    f"Sampling: {active_preset['_data'].get('sampling_interval_minutes', '?')} min  •  "
                    f"Version: {active_preset['_data'].get('version', '?')}  •  "
                    f"Last updated: {active_preset['_data'].get('last_updated', '?')}"
                )
        else:
            active_preset = None  # invalid preset
            st.warning("Selected preset is invalid — proceeding with defaults.")

    # If a preset is active, build auto-map suggestions
    auto_map = {}
    if active_preset and active_preset["_data"]:
        auto_map = auto_map_columns(data.columns.tolist(), active_preset["_data"])

    # ----- Column mapping -----------------------------------------------------

    st.subheader("2. Map your columns")

    # Pre-select timestamp from auto-map if available
    ts_default_idx = 0
    if auto_map.get("timestamp") in data.columns:
        ts_default_idx = data.columns.tolist().index(auto_map["timestamp"])

    col1, col2 = st.columns([1, 2])
    with col1:
        ts_col = st.selectbox(
            "Timestamp column",
            options=data.columns.tolist(),
            index=ts_default_idx,
        )

    # Default parameters: auto-map results if preset is active, else
    # any column matching PARAMETER_CONFIGS keys
    if active_preset and auto_map:
        default_params = [
            auto_map[k] for k in
            ["pH", "temperature", "turbidity", "specific_conductivity",
             "dissolved_oxygen", "stage"]
            if auto_map.get(k) and auto_map[k] in data.columns and auto_map[k] != ts_col
        ]
    else:
        default_params = [c for c in data.columns if c in PARAMETER_CONFIGS]

    with col2:
        param_cols = st.multiselect(
            "Parameter columns to QC",
            options=[c for c in data.columns if c != ts_col],
            default=default_params,
        )

    # ----- Covariate mapping (rainfall + stage) -------------------------------

    st.subheader("2b. Covariates (rainfall + stage)")
    st.caption(
        "Optional but powerful. Rainfall and stage are used to (1) suppress "
        "false-positive flags during real hydrologic events and (2) inform "
        "correction estimates via regression."
    )

    cv1, cv2 = st.columns(2)
    # Determine preset defaults for covariates
    if active_preset and active_preset["_data"]:
        _cov_defaults = active_preset["_data"].get("covariates", {})
        _rain_default = auto_map.get("rainfall") if auto_map.get("rainfall") in data.columns else None
        _stage_default = auto_map.get("stage") if auto_map.get("stage") in data.columns else None
        _rain_win_default = float(_cov_defaults.get("rain_window_hr", 1.0))
        _rain_thr_default = float(_cov_defaults.get("rain_event_threshold", 0.05))
        _stage_q_default = float(_cov_defaults.get("stage_high_quantile", 0.90))
    else:
        _rain_default = "rainfall" if "rainfall" in data.columns else None
        _stage_default = "stage" if "stage" in data.columns else None
        _rain_win_default = 1.0
        _rain_thr_default = 0.05
        _stage_q_default = 0.90

    _rain_options = ["— none —"] + [c for c in data.columns if c != ts_col]
    _stage_options = ["— none —"] + [c for c in data.columns if c != ts_col]

    with cv1:
        rainfall_col = st.selectbox(
            "Rainfall column (e.g. inches per timestep)",
            options=_rain_options,
            index=_rain_options.index(_rain_default) if _rain_default in _rain_options else 0,
        )
        rain_window_hr = st.number_input(
            "Rolling window for rain events (hr)",
            min_value=0.25, max_value=24.0,
            value=_rain_win_default, step=0.25,
        )
        rain_event_threshold = st.number_input(
            "Rain total over window to call it an 'event' (same units as rainfall col)",
            min_value=0.0, value=_rain_thr_default, step=0.01, format="%.3f",
        )
    with cv2:
        stage_col = st.selectbox(
            "Stage column (PT or radar, ft/m)",
            options=_stage_options,
            index=_stage_options.index(_stage_default) if _stage_default in _stage_options else 0,
        )
        stage_high_quantile = st.slider(
            "Stage quantile considered 'high stage'",
            min_value=0.50, max_value=0.99,
            value=_stage_q_default, step=0.01,
        )

    # Convert "none" sentinel to None
    rainfall_col = None if rainfall_col == "— none —" else rainfall_col
    stage_col = None if stage_col == "— none —" else stage_col

    # Warn if a selected covariate is mostly NaN at WQ timestamps
    def _check_covariate_density(col_name, label):
        if col_name and col_name in data.columns:
            pct_valid = data[col_name].notna().mean() * 100
            if pct_valid < 50:
                st.warning(
                    f"⚠️ **{label} column `{col_name}` is {100 - pct_valid:.0f}% empty** "
                    f"at your WQ timestamps. The {label.lower()} feature won't "
                    "contribute meaningfully to event suppression or correction. "
                    "Consider running auto-resample (above) or uploading the "
                    f"{label.lower()} data as a separate file."
                )
    _check_covariate_density(rainfall_col, "Rainfall")
    _check_covariate_density(stage_col, "Stage")

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


    # If a preset is active, build a map from (actual column name) -> ParameterConfig
    preset_configs_by_col: dict[str, ParameterConfig] = {}
    if active_preset and active_preset["_data"]:
        preset_canon_configs = configs_from_preset(active_preset["_data"])
        # auto_map: canonical -> actual column. Invert it.
        canon_to_col = {k: v for k, v in auto_map.items() if v}
        for canon, cfg in preset_canon_configs.items():
            actual_col = canon_to_col.get(canon)
            if actual_col:
                # Re-create config with the actual column name as `name` so it
                # round-trips through the existing per-param loop unchanged
                cfg_dict = {**asdict(cfg), "name": actual_col}
                preset_configs_by_col[actual_col] = ParameterConfig(**cfg_dict)

    custom_configs: dict[str, ParameterConfig] = {}
    for p in param_cols:
        # Priority: preset > built-in default > blank
        default = preset_configs_by_col.get(p) or PARAMETER_CONFIGS.get(p) or ParameterConfig(name=p)
        preset_tag = " (from preset)" if p in preset_configs_by_col else ""
        with st.expander(f"⚙️ {p} ({default.units or 'no units'}){preset_tag}"):
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


# ===========================================================================
# LSTM TRAINING TAB
# ===========================================================================

with tab_train:
    _backend_label = {
        "sklearn": "🌳 Gradient Boosting",
        "tensorflow_lstm": "🧠 LSTM",
    }.get(_ML_BACKEND, "ML")
    st.subheader(f"Train a {_backend_label} model per parameter (PyHydroQC-style)")

    if not _ML_AVAILABLE:
        st.error(
            "**No ML backend installed.** The Train tab needs either "
            "**scikit-learn** (recommended, light) or **TensorFlow** (heavier, "
            "more powerful).\n\n"
            "**Easiest fix — install scikit-learn:**\n"
            "```\npip install scikit-learn\n```\n"
            "This works on any Python 3.9+ and runs fast on CPU.\n\n"
            "**Alternative — TensorFlow LSTM (heavier):**\n"
            "```\npip install -r requirements-lstm.txt\nexport WQC_USE_LSTM=1\n```\n"
            "Needs Python 3.10–3.12. On Streamlit Cloud, pick Python 3.11 "
            "in Advanced settings at deploy time.\n\n"
            f"Detail: `{_ML_BACKEND_ERROR}`"
        )
    else:
        if _ML_BACKEND == "sklearn":
            st.caption(
                "Using **scikit-learn HistGradientBoosting** — fast (seconds), "
                "lightweight, installs anywhere. Trains a **forecast** model "
                "(predicts next clean value from past + covariates → residuals "
                "flag anomalies) and a **correction** model (maps raw windows "
                "to clean values → fills flagged gaps). Saved models reuse "
                "without retraining."
            )
        else:
            st.caption(
                "Using **TensorFlow LSTM**. Trains a **forecast** model "
                "(predicts next clean value from past + covariates → residuals "
                "flag anomalies) and a **correction** model (maps raw windows "
                "to clean values → fills flagged gaps). Saved models reuse "
                "without retraining."
            )

        st.markdown("**Step 1.** Upload your *clean* (corrected) dataset. "
                    "Timestamps must match the raw data already loaded above.")

        clean_file = st.file_uploader(
            "Clean / corrected CSV",
            type=["csv", "xlsx", "xls"], key="clean_upload_train",
        )

        if clean_file is not None:
            try:
                clean_df, clean_note = read_csv_resilient(
                    clean_file, "clean dataset"
                )
                st.success(
                    f"Loaded clean dataset: {len(clean_df):,} rows ({clean_note})."
                )
                clean_ts = detect_timestamp_column(clean_df)
                if clean_ts is None:
                    st.error("Couldn't auto-detect a timestamp column in the clean file.")
                    st.stop()
                clean_df[clean_ts] = pd.to_datetime(clean_df[clean_ts])
                # Align clean to raw timestamps (should be identical, but in case)
                clean_df = clean_df.set_index(clean_ts).reindex(
                    pd.to_datetime(data[ts_col]).values, method="nearest"
                ).reset_index(drop=True)

                st.markdown("**Step 2.** Pick parameter(s) to train.")
                trainable = [
                    c for c in param_cols
                    if c in clean_df.columns and c in data.columns
                ]
                if not trainable:
                    st.warning(
                        "No matching parameter columns between raw and clean data. "
                        f"Raw has: {param_cols}. Clean has: {list(clean_df.columns)}"
                    )
                    st.stop()

                params_to_train = st.multiselect(
                    "Parameters to train",
                    options=trainable, default=trainable,
                )

                st.markdown("**Step 3.** Training settings.")
                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    window_size = st.number_input(
                        "Window size (samples of history)",
                        min_value=8, max_value=512, value=96, step=8,
                        help="96 samples = 24 hours at 15-min sampling",
                    )
                with t2:
                    if _ML_BACKEND == "sklearn":
                        iterations_label = "Max iterations"
                        iterations_default = 200
                        iterations_help = ("Gradient-boosting iterations. Training "
                                           "auto-stops earlier if validation loss plateaus.")
                    else:
                        iterations_label = "Epochs"
                        iterations_default = 50
                        iterations_help = "Number of training passes over the data."
                    iterations = st.number_input(
                        iterations_label,
                        min_value=5, max_value=500,
                        value=iterations_default, step=5,
                        help=iterations_help,
                    )
                with t3:
                    if _ML_BACKEND == "sklearn":
                        # Tree depth controls model capacity for gradient boosting
                        capacity = st.number_input(
                            "Tree max depth",
                            min_value=2, max_value=12, value=6, step=1,
                            help="Controls model capacity. 6 is a good default.",
                        )
                    else:
                        capacity = st.number_input(
                            "LSTM units / layer",
                            min_value=8, max_value=256, value=64, step=8,
                        )
                with t4:
                    threshold_k = st.number_input(
                        "Anomaly threshold k (mean+k*std)",
                        min_value=1.0, max_value=10.0, value=4.0, step=0.5,
                    )

                use_covars_in_lstm = st.checkbox(
                    "Include rainfall + stage as features",
                    value=(rainfall_col is not None or stage_col is not None),
                    disabled=(rainfall_col is None and stage_col is None),
                )

                model_dir = st.text_input(
                    "Save models to folder",
                    value="./models",
                    help="One subfolder per parameter will be created here.",
                )

                if st.button("🚂 Train models", type="primary",
                             use_container_width=True):
                    if not params_to_train:
                        st.error("Pick at least one parameter.")
                        st.stop()

                    # Build covariate frame
                    if use_covars_in_lstm:
                        cov_cols = [c for c in [rainfall_col, stage_col] if c is not None]
                        covar_df = data[cov_cols].copy() if cov_cols else None
                    else:
                        covar_df = None

                    progress = st.progress(0.0, text="Starting...")
                    status = st.empty()

                    trained = {}
                    for i, p in enumerate(params_to_train):
                        clean_series = clean_df[p].reset_index(drop=True)
                        raw_series = data[p].reset_index(drop=True)
                        tol = DEFAULT_LABEL_TOLERANCES.get(p, 0.0)

                        # Build a config using only fields the active backend supports.
                        # Both LearnedConfig and LSTMConfig share these core fields:
                        common_kwargs = dict(
                            parameter=p,
                            window_size=int(window_size),
                            threshold_k=float(threshold_k),
                            label_tolerance=float(tol),
                        )
                        if _ML_BACKEND == "sklearn":
                            cfg = ParameterConfig_ML(
                                max_iter=int(iterations),
                                max_depth=int(capacity),
                                **common_kwargs,
                            )
                        else:
                            cfg = ParameterConfig_ML(
                                epochs=int(iterations),
                                lstm_units=int(capacity),
                                **common_kwargs,
                            )

                        def cb(phase, ep, total, logs):
                            overall = (
                                i + (0.5 if phase == "correction" else 0.0)
                                + (ep / total) * 0.5
                            ) / len(params_to_train)
                            progress.progress(
                                overall,
                                text=f"{p} [{phase}] epoch {ep}/{total}  "
                                     f"loss={logs.get('loss', 0):.4f}",
                            )

                        status.info(f"Training {p}...")
                        lstm = ParameterModel(cfg)
                        lstm.fit(
                            clean=clean_series, raw=raw_series,
                            covariates=covar_df,
                            progress_callback=cb,
                        )
                        save_path = Path(model_dir) / p
                        lstm.save(save_path)
                        trained[p] = str(save_path)

                    progress.progress(1.0, text="Done.")
                    status.success(
                        f"✅ Trained {len(trained)} model(s). "
                        f"Saved to `{model_dir}`."
                    )
                    st.session_state.trained_models = trained
                    st.session_state.lstm_clean_df = clean_df

                    # Show loss curves
                    st.markdown("#### Training history")
                    _x_label = "iteration" if _ML_BACKEND == "sklearn" else "epoch"
                    _y_label = "score" if _ML_BACKEND == "sklearn" else "MSE loss"
                    for p, path in trained.items():
                        lstm = ParameterModel.load(path)
                        fig, ax = plt.subplots(figsize=(10, 2.5))
                        hist = lstm.history.get("forecast")
                        if hist:
                            ax.plot(hist["loss"], label="forecast train", lw=1)
                            if hist["val_loss"]:
                                ax.plot(hist["val_loss"], label="forecast val",
                                        lw=1, ls="--")
                        hist_c = lstm.history.get("correction")
                        if hist_c:
                            ax.plot(hist_c["loss"], label="correction train",
                                    lw=1, color="darkgreen")
                            if hist_c["val_loss"]:
                                ax.plot(hist_c["val_loss"], label="correction val",
                                        lw=1, ls="--", color="darkgreen")
                        ax.set_title(f"{p} — training history")
                        ax.set_xlabel(_x_label)
                        ax.set_ylabel(_y_label)
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

            except Exception as e:
                st.error(f"Couldn't read or process the clean dataset: {e}")


# ===========================================================================
# LSTM DETECT & VALIDATE TAB
# ===========================================================================

with tab_lstm:
    _backend_label2 = {
        "sklearn": "Gradient Boosting",
        "tensorflow_lstm": "LSTM",
    }.get(_ML_BACKEND, "trained")
    st.subheader(f"Run a trained {_backend_label2} model on the raw data")

    if not _ML_AVAILABLE:
        st.error("No ML backend installed — see the Train tab for setup options.")
    else:
        st.caption(
            f"Apply trained {_backend_label2} models to flag anomalies and "
            "(optionally) correct flagged points. If you also have clean reference "
            "data loaded in the Train tab, you'll see precision / recall / F1 metrics."
        )

        model_root = st.text_input(
            "Folder containing trained models",
            value="./models",
            key="lstm_detect_root",
        )

        if Path(model_root).exists():
            available = [d.name for d in Path(model_root).iterdir() if d.is_dir()
                         and (d / "meta.json").exists()]
        else:
            available = []

        if not available:
            st.info(
                f"No trained models found in `{model_root}`. Train some in the "
                "**Train LSTM** tab first."
            )
        else:
            run_params = st.multiselect(
                "Models to apply",
                options=available,
                default=[p for p in available if p in param_cols],
            )

            apply_correction = st.checkbox(
                "Apply correction LSTM to flagged points", value=True,
            )

            if st.button("🔬 Run LSTM detection", type="primary",
                         use_container_width=True):
                if not run_params:
                    st.error("Pick at least one model to apply.")
                    st.stop()

                # Build covariate frame matching what models expect
                covar_df = None
                cov_cols = [c for c in [rainfall_col, stage_col] if c is not None]
                if cov_cols:
                    covar_df = data[cov_cols].copy()

                lstm_results = {}
                metrics_table = []

                for p in run_params:
                    lstm = ParameterModel.load(Path(model_root) / p)
                    raw_series = data[p].reset_index(drop=True)

                    flags, residual, threshold = lstm.detect_anomalies(
                        raw_series, covar_df,
                    )
                    corrected = None
                    if apply_correction and lstm.correction_model is not None:
                        corrected = lstm.correct(raw_series, covar_df)

                    lstm_results[p] = {
                        "flags": flags,
                        "residual": residual,
                        "threshold": threshold,
                        "corrected": corrected,
                        "model": lstm,
                    }

                    # Validation against clean data if available
                    clean_df_state = st.session_state.get("lstm_clean_df")
                    if clean_df_state is not None and p in clean_df_state.columns:
                        truth = derive_labels(
                            raw_series,
                            clean_df_state[p].reset_index(drop=True),
                            tolerance=lstm.config.label_tolerance,
                        )
                        m = compute_metrics(flags, truth)
                        m["parameter"] = p
                        metrics_table.append(m)

                st.session_state.lstm_results = lstm_results

                # ---- Plots
                st.markdown("#### Detection plots")
                ts = data[ts_col]
                for p, r in lstm_results.items():
                    fig, (ax1, ax2) = plt.subplots(
                        2, 1, figsize=(12, 5), sharex=True,
                        gridspec_kw={"height_ratios": [2, 1]},
                    )

                    raw_series = data[p]
                    ax1.plot(ts, raw_series, lw=0.7, color="#1f77b4", label="raw")
                    if r["corrected"] is not None:
                        ax1.plot(ts, r["corrected"], lw=0.7, color="#2ca02c",
                                 alpha=0.7, label="LSTM corrected")
                    if r["flags"].any():
                        mask = r["flags"]
                        ax1.scatter(
                            ts[mask], raw_series[mask],
                            color="red", s=14, label="LSTM flagged", zorder=3,
                        )
                    ax1.set_title(f"{p}: LSTM detection")
                    ax1.legend(fontsize=8, loc="upper right")
                    ax1.grid(alpha=0.3)

                    ax2.plot(ts, r["residual"], lw=0.6, color="grey",
                             label="|residual|")
                    ax2.plot(ts, r["threshold"], lw=0.8, color="red",
                             label="dynamic threshold")
                    ax2.set_ylabel("residual")
                    ax2.legend(fontsize=8, loc="upper right")
                    ax2.grid(alpha=0.3)

                    st.pyplot(fig)
                    plt.close(fig)

                # ---- Metrics
                if metrics_table:
                    st.markdown("#### Validation metrics "
                                "(vs. derived ground truth from clean data)")
                    mdf = pd.DataFrame(metrics_table)
                    cols_order = ["parameter", "precision", "recall", "f1",
                                  "accuracy", "true_positives", "false_positives",
                                  "false_negatives", "true_negatives"]
                    mdf = mdf[[c for c in cols_order if c in mdf.columns]]
                    st.dataframe(mdf, use_container_width=True, hide_index=True)
                    st.caption(
                        "Higher precision = fewer false alarms. "
                        "Higher recall = fewer missed anomalies. "
                        "F1 balances both."
                    )

                # ---- Download
                st.markdown("#### Download LSTM results")
                out_df = data.copy()
                for p, r in lstm_results.items():
                    out_df[f"{p}_lstm_flag"] = r["flags"].values
                    out_df[f"{p}_lstm_residual"] = r["residual"].values
                    if r["corrected"] is not None:
                        out_df[f"{p}_lstm_corrected"] = r["corrected"].values

                buf = io.BytesIO()
                out_df.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ LSTM-flagged + corrected CSV",
                    data=buf.getvalue(),
                    file_name="lstm_qc_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
