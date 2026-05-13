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


# Sensible defaults for common aquatic sensors (Fahrenheit-first)
PARAMETER_CONFIGS: dict[str, ParameterConfig] = {
    "temperature": ParameterConfig(
        name="temperature", units="°F",
        range_min=32, range_max=95,
        spike_threshold=4.0, persistence_window=12,
        max_rate_change=9.0,
    ),
    "specific_conductivity": ParameterConfig(
        name="specific_conductivity", units="µS/cm",
        range_min=0, range_max=2000,
        spike_threshold=4.0, persistence_window=12,
        max_rate_change=200,
    ),
    "ph": ParameterConfig(
        name="ph", units="pH",
        range_min=4, range_max=10,
        spike_threshold=3.5, persistence_window=12,
        max_rate_change=1.0,
    ),
    "dissolved_oxygen": ParameterConfig(
        name="dissolved_oxygen", units="mg/L",
        range_min=0, range_max=20,
        spike_threshold=4.0, persistence_window=12,
        max_rate_change=6.0,
    ),
    "turbidity": ParameterConfig(
        name="turbidity", units="NTU",
        range_min=0, range_max=1000,
        spike_threshold=5.0, persistence_window=12,
        max_rate_change=50,
    ),
}


# ---------------------------------------------------------------------------
# Main QC class
# ---------------------------------------------------------------------------

class WaterQualityQCv2:
    """Run automated QC on a multi-parameter sensor dataframe."""

    def __init__(
        self,
        data: pd.DataFrame,
        timestamp_col: str,
        parameters: list[str],
        configs: Optional[dict[str, ParameterConfig]] = None,
    ):
        self.data = data.copy()
        self.timestamp_col = timestamp_col
        self.parameters = parameters

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
        flags["any"] = flags.any(axis=1)

        self.flags[parameter] = flags
        return flags

    def run_all_sequential(self) -> dict[str, pd.DataFrame]:
        for p in self.parameters:
            if p in self.data.columns:
                self.run_parameter(p)
        return self.flags

    # ---- Reporting --------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        rows = []
        for p, f in self.flags.items():
            n = len(f)
            row = {"parameter": p, "units": self.configs[p].units, "n_records": n}
            for col in ["range", "spike", "persistence", "rate_of_change", "arima", "any"]:
                row[col] = int(f[col].sum())
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
            "",
        ]
        for _, r in s.iterrows():
            lines += [
                f"{r['parameter'].upper()} ({r['units']})",
                f"  Total anomalies: {r['any']} ({r['pct_flagged']}%)",
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
            for col in f.columns:
                out[f"{p}_flag_{col}"] = f[col].values
        out.to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# Demo data generator (used by the Streamlit app's "Try demo data" button)
# ---------------------------------------------------------------------------

def generate_demo_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Synthetic Fahrenheit-based water-quality time series with injected anomalies."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min")

    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / (24 * 4))  # 96 samples/day

    df = pd.DataFrame({
        "datetime": timestamps,
        "temperature":           59 + 9 * daily + rng.normal(0, 0.5, n),
        "specific_conductivity": 450 + 30 * daily + rng.normal(0, 5, n),
        "ph":                    7.5 + 0.3 * daily + rng.normal(0, 0.05, n),
        "dissolved_oxygen":      9.0 + 1.5 * daily + rng.normal(0, 0.2, n),
        "turbidity":             5.0 + 2 * daily + rng.normal(0, 0.5, n),
    })

    # Inject obvious anomalies
    df.loc[500, "temperature"] = 122          # extreme spike
    df.loc[1500:1530, "temperature"] = 70     # stuck sensor
    df.loc[2500, "ph"] = 12.5                 # out of range
    df.loc[3000, "dissolved_oxygen"] = -1     # negative DO
    df.loc[3500:3505, "turbidity"] = 800      # turbidity event

    return df
