"""
Water Quality QC Framework v2
==============================
Automated quality control for aquatic sensor data.

Detection methods:
- Range checks (physically plausible bounds)
- Spike detection (z-score on first differences)
- Persistence (stuck sensor / flatline)
- Rate-of-change (unrealistic jumps)
- ARIMA forecast residuals (optional, statsmodels)

All temperature configs are in °F by default. DO saturation internally
converts °F → °C before applying the Garcia-Gordon equation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Parameter configuration
# ---------------------------------------------------------------------------

@dataclass
class ParameterConfig:
    """Per-parameter QC settings.

    Note: defaults use large finite sentinels rather than np.inf, because
    Streamlit's number_input widget cannot serialize infinity to JavaScript.
    A range_min of -1e9 is effectively "no lower bound" for any real sensor.
    """
    name: str
    units: str = ""
    range_min: float = -1e9
    range_max: float = 1e9
    spike_threshold: float = 4.0          # z-score on first differences
    persistence_window: int = 6           # consecutive identical readings
    persistence_tol: float = 1e-6
    max_rate_change: float = 1e9          # per hour (effectively unlimited)
    use_arima: bool = False
    arima_order: tuple = (2, 0, 1)
    arima_threshold: float = 3.5

    # ---- Event-aware flagging (set per-parameter; sensible defaults below)
    # When True, suppress these flag types during a rain event.
    suppress_spike_in_rain: bool = False
    suppress_rate_in_rain: bool = False
    suppress_range_min_in_high_stage: bool = False  # e.g. real hypoxia during floods
    # When True, use covariates (rain, stage) in correction regression.
    use_covariates_for_correction: bool = False


# Sensible defaults for common aquatic sensors (Fahrenheit-first)
PARAMETER_CONFIGS: dict[str, ParameterConfig] = {
    "temperature": ParameterConfig(
        name="temperature", units="°F",
        range_min=32, range_max=95,
        spike_threshold=4.0, persistence_window=12,
        max_rate_change=9.0,
        # Temp doesn't respond fast to storms — no suppression
        use_covariates_for_correction=False,
    ),
    "specific_conductivity": ParameterConfig(
        name="specific_conductivity", units="µS/cm",
        range_min=0, range_max=2000,
        spike_threshold=4.0, persistence_window=12,
        max_rate_change=200,
        # Rain dilutes conductivity rapidly — suppress rate flags during rain
        suppress_rate_in_rain=True,
        use_covariates_for_correction=True,
    ),
    "ph": ParameterConfig(
        name="ph", units="pH",
        range_min=4, range_max=10,
        spike_threshold=3.5, persistence_window=12,
        max_rate_change=1.0,
        # Acid pulses from runoff are real
        suppress_rate_in_rain=True,
        use_covariates_for_correction=True,
    ),
    "dissolved_oxygen": ParameterConfig(
        name="dissolved_oxygen", units="mg/L",
        range_min=0, range_max=20,
        spike_threshold=4.0, persistence_window=12,
        max_rate_change=6.0,
        # Floods cause real low-DO events
        suppress_range_min_in_high_stage=True,
        suppress_rate_in_rain=True,
        use_covariates_for_correction=True,
    ),
    "turbidity": ParameterConfig(
        name="turbidity", units="NTU",
        range_min=0, range_max=1000,
        spike_threshold=5.0, persistence_window=12,
        max_rate_change=50,
        # Sediment pulses during rain are real, not anomalies
        suppress_spike_in_rain=True,
        suppress_rate_in_rain=True,
        use_covariates_for_correction=True,
    ),
}


# ---------------------------------------------------------------------------
# Main QC class
# ---------------------------------------------------------------------------

class WaterQualityQCv2:
    """Run automated QC on a multi-parameter sensor dataframe.

    Optional covariates:
      rainfall_col   : column name for rainfall (per timestep, e.g. inches/5-min)
      stage_col      : column name for stage (PT or radar, ft or m)

    Event thresholds (control when "rain event" / "high stage" are True):
      rain_window_hr      : hours over which to sum rainfall (default 1.0)
      rain_event_threshold: cumulative rainfall over the window above which we
                            consider it raining (default 0.05 inches)
      stage_high_quantile : stage above this quantile is "high stage"
                            (default 0.90)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        timestamp_col: str,
        parameters: list[str],
        configs: Optional[dict[str, ParameterConfig]] = None,
        rainfall_col: Optional[str] = None,
        stage_col: Optional[str] = None,
        rain_window_hr: float = 1.0,
        rain_event_threshold: float = 0.05,
        stage_high_quantile: float = 0.90,
    ):
        self.data = data.copy()
        self.timestamp_col = timestamp_col
        self.parameters = parameters
        self.rainfall_col = rainfall_col
        self.stage_col = stage_col
        self.rain_window_hr = rain_window_hr
        self.rain_event_threshold = rain_event_threshold
        self.stage_high_quantile = stage_high_quantile

        # Parse timestamps and sort
        self.data[timestamp_col] = pd.to_datetime(self.data[timestamp_col])
        self.data = self.data.sort_values(timestamp_col).reset_index(drop=True)

        # Merge user configs over defaults
        self.configs: dict[str, ParameterConfig] = {}
        for p in parameters:
            if configs and p in configs:
                self.configs[p] = configs[p]
            elif p in PARAMETER_CONFIGS:
                self.configs[p] = PARAMETER_CONFIGS[p]
            else:
                self.configs[p] = ParameterConfig(name=p)

        # Storage for flags: one DataFrame per parameter
        self.flags: dict[str, pd.DataFrame] = {}
        # Storage for corrected series: one Series per parameter
        self.corrected: dict[str, pd.Series] = {}

        # Compute event indicators once, up front
        self._compute_event_indicators()

    # ---- Event detection -------------------------------------------------

    def _compute_event_indicators(self) -> None:
        """Derive rain_event and high_stage boolean masks from covariates."""
        n = len(self.data)
        self.rain_event = pd.Series(False, index=self.data.index)
        self.high_stage = pd.Series(False, index=self.data.index)

        if self.rainfall_col and self.rainfall_col in self.data.columns:
            rain = self.data[self.rainfall_col].fillna(0)
            # Detect sample frequency to size the rolling window
            dt = self.data[self.timestamp_col].diff().median()
            if pd.isna(dt) or dt.total_seconds() == 0:
                samples_per_window = 12  # fallback
            else:
                samples_per_window = max(
                    1,
                    int(round(self.rain_window_hr * 3600 / dt.total_seconds())),
                )
            rolling = rain.rolling(samples_per_window, min_periods=1).sum()
            self.rain_event = rolling >= self.rain_event_threshold
            self.data["_rain_rolling"] = rolling

        if self.stage_col and self.stage_col in self.data.columns:
            stage = self.data[self.stage_col]
            cutoff = stage.quantile(self.stage_high_quantile)
            self.high_stage = stage >= cutoff
            self.data["_stage_cutoff"] = cutoff

    # ---- Detection methods ------------------------------------------------

    def _flag_range(self, series: pd.Series, cfg: ParameterConfig) -> pd.Series:
        return (series < cfg.range_min) | (series > cfg.range_max)

    def _flag_spike(self, series: pd.Series, cfg: ParameterConfig) -> pd.Series:
        diff = series.diff()
        mean, std = diff.mean(), diff.std()
        if std == 0 or np.isnan(std):
            return pd.Series(False, index=series.index)
        z = (diff - mean).abs() / std
        return z > cfg.spike_threshold

    def _flag_persistence(self, series: pd.Series, cfg: ParameterConfig) -> pd.Series:
        flat = series.diff().abs() < cfg.persistence_tol
        # Mark runs of length >= persistence_window
        groups = (~flat).cumsum()
        run_lengths = flat.groupby(groups).transform("sum")
        return flat & (run_lengths >= cfg.persistence_window)

    def _flag_rate(self, series: pd.Series, cfg: ParameterConfig) -> pd.Series:
        if cfg.max_rate_change >= 1e9:
            return pd.Series(False, index=series.index)
        dt_hours = (
            self.data[self.timestamp_col]
            .diff().dt.total_seconds() / 3600
        )
        rate = (series.diff() / dt_hours).abs()
        return rate > cfg.max_rate_change

    def _flag_arima(self, series: pd.Series, cfg: ParameterConfig) -> pd.Series:
        if not cfg.use_arima:
            return pd.Series(False, index=series.index)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            clean = series.dropna()
            if len(clean) < 50:
                return pd.Series(False, index=series.index)
            model = ARIMA(clean, order=cfg.arima_order).fit()
            resid = model.resid
            z = (resid - resid.mean()).abs() / resid.std()
            flagged = pd.Series(False, index=series.index)
            flagged.loc[clean.index] = z > cfg.arima_threshold
            return flagged
        except Exception:
            return pd.Series(False, index=series.index)

    # ---- Orchestration ----------------------------------------------------

    def run_parameter(self, parameter: str) -> pd.DataFrame:
        cfg = self.configs[parameter]
        series = self.data[parameter]

        flags = pd.DataFrame({
            "range": self._flag_range(series, cfg),
            "spike": self._flag_spike(series, cfg),
            "persistence": self._flag_persistence(series, cfg),
            "rate_of_change": self._flag_rate(series, cfg),
            "arima": self._flag_arima(series, cfg),
        }, index=series.index)

        # Capture raw (pre-suppression) flags for transparency
        flags_raw = flags.copy()

        # ---- Event-aware suppression ----
        suppressed = pd.DataFrame(False, index=flags.index, columns=flags.columns)
        if cfg.suppress_spike_in_rain and self.rain_event.any():
            mask = flags["spike"] & self.rain_event
            suppressed["spike"] = mask
            flags.loc[mask, "spike"] = False
        if cfg.suppress_rate_in_rain and self.rain_event.any():
            mask = flags["rate_of_change"] & self.rain_event
            suppressed["rate_of_change"] = mask
            flags.loc[mask, "rate_of_change"] = False
        if cfg.suppress_range_min_in_high_stage and self.high_stage.any():
            # Only suppress low-end range violations, not high
            below_min = series < cfg.range_min
            mask = flags["range"] & self.high_stage & below_min
            suppressed["range"] = mask
            flags.loc[mask, "range"] = False

        flags["any"] = flags.any(axis=1)
        flags["suppressed"] = suppressed.any(axis=1)
        flags["rain_event"] = self.rain_event.values
        flags["high_stage"] = self.high_stage.values
        # Keep raw flags side-by-side for audit
        for col in ["range", "spike", "rate_of_change"]:
            flags[f"{col}_raw"] = flags_raw[col]

        self.flags[parameter] = flags
        return flags

    def run_all_sequential(self) -> dict[str, pd.DataFrame]:
        for p in self.parameters:
            if p in self.data.columns:
                self.run_parameter(p)
        return self.flags

    # ---- Correction -------------------------------------------------------

    def correct_parameter(
        self,
        parameter: str,
        max_gap_for_interp: int = 4,
    ) -> pd.Series:
        """Build a corrected series for one parameter.

        Strategy (in order):
          1. Start with the raw series.
          2. Set flagged points to NaN.
          3. For short gaps (<= max_gap_for_interp samples), linear interpolate.
          4. For longer gaps, if covariates are enabled, predict from a
             rainfall/stage regression fit on the un-flagged surroundings.
             Otherwise, linear interpolate the remainder anyway.
          5. Cross-fade the regression prediction with linear interpolation
             at the gap edges so corrected values join smoothly.
        """
        if parameter not in self.flags:
            raise ValueError(f"Run QC for '{parameter}' before correcting.")

        cfg = self.configs[parameter]
        flags = self.flags[parameter]
        raw = self.data[parameter].astype(float).copy()

        # Step 1-2: blank out flagged points
        working = raw.copy()
        working[flags["any"]] = np.nan

        # Step 3: short-gap linear interpolation
        working = working.interpolate(
            method="linear", limit=max_gap_for_interp, limit_area="inside",
        )

        # Step 4: longer gaps - covariate regression (if enabled and possible)
        still_missing = working.isna() & flags["any"]
        if cfg.use_covariates_for_correction and still_missing.any():
            predicted = self._covariate_predict(parameter, raw, flags["any"])
            if predicted is not None:
                # Cross-fade: at gap edges, blend with linear interp for continuity
                blended = self._cross_fade(working, predicted, still_missing)
                working = working.where(~still_missing, blended)

        # Step 5: any remaining NaNs get a final linear interpolation
        working = working.interpolate(method="linear", limit_area="inside")
        # And forward/backward fill at the very edges if needed
        working = working.ffill().bfill()

        self.corrected[parameter] = working
        return working

    def _covariate_predict(
        self,
        parameter: str,
        raw: pd.Series,
        flagged: pd.Series,
    ) -> Optional[pd.Series]:
        """Fit a simple linear regression of the parameter on available
        covariates using un-flagged points; return predictions for all rows.

        Returns None if no covariates are available or fit fails.
        """
        X_parts = []
        if self.rainfall_col and self.rainfall_col in self.data.columns:
            rain = self.data[self.rainfall_col].fillna(0)
            X_parts.append(("rain", rain))
            if "_rain_rolling" in self.data.columns:
                X_parts.append(("rain_roll", self.data["_rain_rolling"]))
        if self.stage_col and self.stage_col in self.data.columns:
            stage = self.data[self.stage_col]
            X_parts.append(("stage", stage))
            X_parts.append(("stage_diff", stage.diff().fillna(0)))

        if not X_parts:
            return None

        # Build design matrix
        X_df = pd.DataFrame({name: vals.values for name, vals in X_parts},
                            index=raw.index).astype(float)
        X_df = X_df.fillna(X_df.median(numeric_only=True))

        # Fit on un-flagged points only
        y = raw.copy()
        train_mask = (~flagged) & y.notna()
        if train_mask.sum() < max(20, 4 * X_df.shape[1]):
            return None  # not enough clean data to fit

        X_train = X_df.loc[train_mask].values
        y_train = y.loc[train_mask].values

        # Least-squares with intercept
        try:
            X_aug = np.column_stack([np.ones(len(X_train)), X_train])
            coefs, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
            X_all_aug = np.column_stack([np.ones(len(X_df)), X_df.values])
            preds = X_all_aug @ coefs
            return pd.Series(preds, index=raw.index)
        except Exception:
            return None

    @staticmethod
    def _cross_fade(
        interpolated: pd.Series,
        predicted: pd.Series,
        gap_mask: pd.Series,
        fade_samples: int = 4,
    ) -> pd.Series:
        """Within each gap, weight predicted values heavily in the middle and
        the interpolated/linear-trend values at the edges, so the joins are
        seamless. PyHydroQC-style cross-fade.
        """
        out = predicted.copy()
        # Identify gap groups (runs of True in gap_mask)
        gap_id = (gap_mask != gap_mask.shift()).cumsum()
        for gid, idx in gap_mask.groupby(gap_id).groups.items():
            if not gap_mask.loc[idx].iloc[0]:
                continue
            n = len(idx)
            # Build a triangular weight: 0 at edges -> 1 in middle, over fade_samples
            fade = min(fade_samples, n // 2)
            w = np.ones(n)
            if fade > 0:
                ramp = np.linspace(0, 1, fade + 1)[1:]
                w[:fade] = ramp
                w[-fade:] = ramp[::-1]
            # Linear endpoint anchor: use neighboring non-gap values
            pre_idx = idx[0] - 1
            post_idx = idx[-1] + 1
            if pre_idx in interpolated.index and post_idx in interpolated.index:
                pre_val = interpolated.iloc[pre_idx]
                post_val = interpolated.iloc[post_idx]
                if pd.notna(pre_val) and pd.notna(post_val):
                    linear = np.linspace(pre_val, post_val, n + 2)[1:-1]
                    pred_vals = predicted.loc[idx].values
                    blended = w * pred_vals + (1 - w) * linear
                    out.loc[idx] = blended
        return out

    def correct_all(self, max_gap_for_interp: int = 4) -> dict[str, pd.Series]:
        for p in self.parameters:
            if p in self.flags:
                self.correct_parameter(p, max_gap_for_interp=max_gap_for_interp)
        return self.corrected

    # ---- Reporting --------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        rows = []
        for p, f in self.flags.items():
            n = len(f)
            row = {"parameter": p, "units": self.configs[p].units, "n_records": n}
            for col in ["range", "spike", "persistence", "rate_of_change", "arima", "any"]:
                row[col] = int(f[col].sum())
            row["suppressed_by_events"] = int(f["suppressed"].sum())
            row["pct_flagged"] = round(100 * f["any"].sum() / n, 2) if n else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def write_summary_report(self, path: str | Path) -> Path:
        path = Path(path)
        s = self.summary()
        lines = [
            "WATER QUALITY QC SUMMARY REPORT",
            "=" * 40,
            f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
            f"Total records: {len(self.data):,}",
            f"Date range: {self.data[self.timestamp_col].min()} "
            f"to {self.data[self.timestamp_col].max()}",
        ]
        if self.rainfall_col:
            lines.append(
                f"Rainfall covariate: '{self.rainfall_col}'  "
                f"(rain events: {int(self.rain_event.sum())} samples, "
                f"{round(100*self.rain_event.mean(),1)}%)"
            )
        if self.stage_col:
            lines.append(
                f"Stage covariate: '{self.stage_col}'  "
                f"(high-stage events: {int(self.high_stage.sum())} samples, "
                f"{round(100*self.high_stage.mean(),1)}%)"
            )
        lines.append("")
        for _, r in s.iterrows():
            lines += [
                f"{r['parameter'].upper()} ({r['units']})",
                f"  Total anomalies: {r['any']} ({r['pct_flagged']}%)",
                f"  Suppressed by events: {r['suppressed_by_events']}",
                f"  Detection breakdown:",
                f"    - range:          {r['range']}",
                f"    - spike:          {r['spike']}",
                f"    - persistence:    {r['persistence']}",
                f"    - rate_of_change: {r['rate_of_change']}",
                f"    - arima:          {r['arima']}",
                "",
            ]
        path.write_text("\n".join(lines))
        return path

    def export_flagged_csv(self, path: str | Path) -> Path:
        path = Path(path)
        out = self.data.copy()
        for p, f in self.flags.items():
            for col in ["range", "spike", "persistence", "rate_of_change",
                        "arima", "any", "suppressed"]:
                if col in f.columns:
                    out[f"{p}_flag_{col}"] = f[col].values
            if p in self.corrected:
                out[f"{p}_corrected"] = self.corrected[p].values
        out.to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# Demo data generator (used by the Streamlit app's "Try demo data" button)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-file timestamp alignment
# ---------------------------------------------------------------------------

def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort: pick the column that looks most like a timestamp.

    Strategy:
      1. Column names containing 'time', 'date', 'datetime', 'stamp'
      2. Fall back to first column whose values parse as datetimes
    """
    name_hints = ("datetime", "timestamp", "time", "date")
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in name_hints):
            try:
                pd.to_datetime(df[c].head(50), errors="raise")
                return c
            except Exception:
                continue
    # Fallback: try each column
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c].head(50), errors="raise")
            if parsed.notna().all():
                return c
        except Exception:
            continue
    return None


def align_to_grid(
    target: pd.DataFrame,
    target_ts_col: str,
    external: pd.DataFrame,
    external_ts_col: str,
    value_cols: list[str],
    tolerance_minutes: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Align external (rainfall or stage) data to the target timestamp grid.

    For each row in `target`, find the nearest external observation within
    `tolerance_minutes`. Points outside tolerance become NaN.

    For rainfall (typically a *rate* measured at points), this gives the
    nearest sample. For cumulative rainfall, the caller should difference
    first.

    Returns:
        merged: copy of `target` with the external value columns appended
        diag:   dict with alignment quality metrics
    """
    t = target[[target_ts_col]].copy()
    t[target_ts_col] = pd.to_datetime(t[target_ts_col], errors="coerce")
    # merge_asof requires non-null keys; drop rows with bad timestamps
    t_valid = t.dropna(subset=[target_ts_col]).sort_values(target_ts_col).reset_index()
    t_valid = t_valid.rename(columns={"index": "_orig_idx"})

    e = external.copy()
    e[external_ts_col] = pd.to_datetime(e[external_ts_col], errors="coerce")
    e = e.dropna(subset=[external_ts_col])
    keep = [external_ts_col] + value_cols
    e = e[keep].sort_values(external_ts_col).reset_index(drop=True)

    # merge_asof does nearest-neighbor with a tolerance
    merged_valid = pd.merge_asof(
        t_valid, e,
        left_on=target_ts_col, right_on=external_ts_col,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )

    # Build the full-length result, putting NaN for rows whose target timestamp was bad
    full = target.copy()
    full[target_ts_col] = pd.to_datetime(full[target_ts_col], errors="coerce")
    full = full.sort_values(target_ts_col).reset_index(drop=True)
    for vc in value_cols:
        col = pd.Series(np.nan, index=full.index, dtype="float64")
        # Map valid rows by their original index
        col.iloc[merged_valid["_orig_idx"].values] = merged_valid[vc].values
        full[vc] = col.values

    # Diagnostics
    diag = {
        "target_rows": len(full),
        "target_rows_valid_ts": len(t_valid),
        "external_rows": len(e),
        "tolerance_min": tolerance_minutes,
    }
    for vc in value_cols:
        matched = full[vc].notna().sum()
        diag[f"{vc}_matched"] = int(matched)
        diag[f"{vc}_match_pct"] = round(100 * matched / len(full), 1) if len(full) else 0

    return full, diag


def generate_demo_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Synthetic Fahrenheit water-quality time series with rainfall + stage.

    Includes:
      - Diurnal cycles on temp, DO, pH, SpC, turbidity
      - Two simulated rain events (rainfall + stage rise + turbidity spike +
        SpC dilution + pH drop) — these are REAL responses and should be
        preserved by event-aware QC, not flagged.
      - Injected sensor faults: extreme spikes, stuck sensors, out-of-range.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min")

    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / (24 * 4))  # 96 samples/day

    # Baseline series
    temp = 59 + 9 * daily + rng.normal(0, 0.5, n)
    spc = 450 + 30 * daily + rng.normal(0, 5, n)
    ph = 7.5 + 0.3 * daily + rng.normal(0, 0.05, n)
    do = 9.0 + 1.5 * daily + rng.normal(0, 0.2, n)
    turb = 5.0 + 2 * daily + rng.normal(0, 0.5, n)

    # ---- Rainfall + stage with two real storm events ----
    rainfall = np.zeros(n)   # inches per 15-min step
    stage = 2.5 + 0.05 * daily + rng.normal(0, 0.02, n)  # ft, gentle baseline

    def storm(start: int, peak_in_per_step: float, duration: int):
        # Triangular rainfall pulse
        rise = duration // 3
        for i in range(duration):
            if i < rise:
                rainfall[start + i] += peak_in_per_step * (i / rise)
            else:
                rainfall[start + i] += peak_in_per_step * max(
                    0, 1 - (i - rise) / (duration - rise)
                )
        # Stage rises ~lagging by 6 samples (90 min) and recedes slowly
        lag = 6
        for i in range(duration * 3):
            idx = start + lag + i
            if 0 <= idx < n:
                rise_pct = min(1, i / (duration * 0.6))
                fall = max(0, 1 - (i - duration * 0.6) / (duration * 2))
                stage[idx] += 1.8 * rise_pct * fall

    storm(start=int(n * 0.16), peak_in_per_step=0.08, duration=40)   # ~10 hr event
    storm(start=int(n * 0.44), peak_in_per_step=0.05, duration=30)   # ~7.5 hr event

    # ---- Real WQ responses to storms (NOT anomalies) ----
    rain_roll = pd.Series(rainfall).rolling(12, min_periods=1).sum().values
    stage_anomaly = stage - (2.5 + 0.05 * daily)
    # Turbidity spikes during storms
    turb += 80 * np.clip(rain_roll, 0, 1.5) + 30 * np.clip(stage_anomaly, 0, 2)
    # Specific conductivity dilutes
    spc -= 120 * np.clip(rain_roll, 0, 1.5)
    # pH dips slightly with acidic runoff
    ph -= 0.4 * np.clip(rain_roll, 0, 1.5)
    # DO drops during high-stage (organic load, mixing of anoxic water)
    do -= 2.0 * np.clip(stage_anomaly, 0, 2)

    df = pd.DataFrame({
        "datetime": timestamps,
        "temperature": temp,
        "specific_conductivity": spc,
        "ph": ph,
        "dissolved_oxygen": do,
        "turbidity": turb,
        "rainfall": rainfall,
        "stage": stage,
    })

    # Inject obvious sensor faults (these SHOULD be flagged)
    df.loc[500, "temperature"] = 122            # extreme spike
    df.loc[1500:1530, "temperature"] = 70       # stuck sensor
    df.loc[2500, "ph"] = 12.5                   # out of range
    df.loc[3000, "dissolved_oxygen"] = -1       # negative DO
    df.loc[4500, "turbidity"] = 1500            # impossible turbidity spike OUTSIDE storm

    return df
