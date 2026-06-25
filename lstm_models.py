"""
Learned models for water-quality anomaly detection and correction.
==================================================================
Sklearn-only alternative to lstm_models.py. Same public API, but uses
HistGradientBoostingRegressor with lag features instead of an LSTM.

Why this exists: TensorFlow is hard to install on many systems (no wheels
for Python 3.13/3.14 as of 2026, Windows DLL issues, ~500 MB footprint).
Sklearn installs on every supported Python version and runs anywhere.

Tradeoffs vs LSTM:
  + Trains in seconds instead of minutes (no GPU needed)
  + Installs cleanly on any Python via `pip install scikit-learn`
  + Small model files (~1 MB) instead of large .keras checkpoints
  - Slightly less predictive power on complex sequence patterns
  - Lag features must be hand-engineered (LSTM learns them implicitly)
  - For most water-quality QC tasks the gap is small in practice

Implements the PyHydroQC (Jones 2022) supervised workflow:
    Forecast model: trained on clean data, predicts next clean value from
                    past clean values + covariates. At inference, large
                    residuals on the raw stream = anomalies.
    Correction model: trained on (raw lag window, covariates) -> clean
                      value pairs, used to fill flagged gaps.

Anomaly detection uses a dynamic threshold:
    mean(|residual|) + k * std(|residual|)   over a rolling clean window
followed by Jones' "window widening" to catch full event extents.

Validation: precision / recall / F1 vs. derived ground truth (raw != clean
within tolerance) per Jones step 7.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Sklearn is the only ML dep, and it's lightweight (no TF/Keras/Theano)
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Training configuration  (mirrors LSTMConfig fields where they make sense)
# ---------------------------------------------------------------------------

@dataclass
class LearnedConfig:
    """Per-parameter training settings."""
    parameter: str
    window_size: int = 96            # samples of history per prediction
    lag_summary_count: int = 8       # how many summary stats per window
    max_iter: int = 200              # gradient-boosting iterations
    learning_rate: float = 0.05
    max_depth: int = 6
    validation_fraction: float = 0.15
    random_state: int = 42
    # Anomaly detection
    threshold_k: float = 4.0
    rolling_threshold_window: int = 1000
    widen_window: int = 4
    # Label derivation
    label_tolerance: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "LearnedConfig":
        return cls(**json.loads(s))


# Sensible per-parameter label tolerances (sensor resolution)
DEFAULT_LABEL_TOLERANCES = {
    "temperature": 0.1,
    "specific_conductivity": 1.0,
    "ph": 0.01,
    "dissolved_oxygen": 0.05,
    "turbidity": 0.5,
}


# ---------------------------------------------------------------------------
# Label derivation from raw + clean pair
# ---------------------------------------------------------------------------

def derive_labels(
    raw: pd.Series,
    clean: pd.Series,
    tolerance: float = 0.0,
) -> pd.Series:
    """Boolean label: True where raw differs from clean by more than tolerance."""
    diff = (raw - clean).abs()
    label = diff > tolerance
    label = label.where(raw.notna() & clean.notna(), False)
    return label


# ---------------------------------------------------------------------------
# Feature engineering: turn a windowed time series into tabular features
# ---------------------------------------------------------------------------

def make_lag_features(
    series: np.ndarray,
    window_size: int,
    covariates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a feature matrix where each row summarizes a window of history.

    Features per row (for the target series):
      - value at lag 1, 2, 3, 6, 12, 24 (or up to window_size)
      - mean, std, min, max, median of the window
      - mean of first half, mean of second half (trend signal)
      - covariates at the prediction timestep (if provided)

    This gives the gradient-boosting model enough temporal context to
    rival a small LSTM on most univariate problems.
    """
    n = len(series)
    n_pred = n - window_size
    if n_pred <= 0:
        return np.empty((0, 0))

    # Specific lags (use what's available within window_size)
    lag_indices = [1, 2, 3, 6, 12, 24, 48, 96]
    lag_indices = [l for l in lag_indices if l <= window_size]

    features = []
    for i in range(n_pred):
        end = i + window_size
        win = series[i:end]
        row = []
        # Specific lags (most recent ones first)
        for lag in lag_indices:
            row.append(win[-lag])
        # Window summary stats
        row.extend([
            np.nanmean(win),
            np.nanstd(win),
            np.nanmin(win),
            np.nanmax(win),
            np.nanmedian(win),
            np.nanmean(win[: window_size // 2]),     # first-half mean
            np.nanmean(win[window_size // 2 :]),     # second-half mean
        ])
        # Covariates at the prediction timestep
        if covariates is not None and covariates.size > 0:
            row.extend(covariates[end].tolist())
        features.append(row)
    return np.array(features, dtype=float)


# ---------------------------------------------------------------------------
# Trainer: holds forecast + correction models for one parameter
# ---------------------------------------------------------------------------

class ParameterLearned:
    """Forecast + correction sklearn models for a single parameter.

    Public API matches lstm_models.ParameterLSTM so the Streamlit UI can
    use either interchangeably.
    """

    def __init__(self, config: LearnedConfig):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install with `pip install scikit-learn`."
            )
        self.config = config
        self.forecast_model = None
        self.correction_model = None
        self.scaler_target = StandardScaler()
        self.scaler_covar = StandardScaler()
        self.covar_names: list[str] = []
        self.history = {"forecast": None, "correction": None}

    # ---- Training ---------------------------------------------------------

    def fit(
        self,
        clean: pd.Series,
        raw: Optional[pd.Series] = None,
        covariates: Optional[pd.DataFrame] = None,
        progress_callback=None,
    ) -> dict:
        """Train forecast (always) and correction (if raw provided) models.

        Returns a dict of training history (validation loss curves) per model.
        """
        cfg = self.config

        # ---- Prep target ----
        clean_arr = clean.values.astype(float)
        valid = ~np.isnan(clean_arr)
        self.scaler_target.fit(clean_arr[valid].reshape(-1, 1))

        # ---- Prep covariates ----
        if covariates is not None and len(covariates.columns) > 0:
            self.covar_names = list(covariates.columns)
            cov_arr = covariates.values.astype(float)
            for j in range(cov_arr.shape[1]):
                m = np.nanmedian(cov_arr[:, j])
                cov_arr[np.isnan(cov_arr[:, j]), j] = m
            self.scaler_covar.fit(cov_arr)
            cov_scaled = self.scaler_covar.transform(cov_arr)
        else:
            cov_scaled = None

        clean_scaled = self.scaler_target.transform(clean_arr.reshape(-1, 1)).ravel()
        clean_scaled_filled = np.where(valid, clean_scaled, 0.0)

        # ---- Forecast model: predict next clean from past clean + covariates ----
        if progress_callback:
            progress_callback("forecast", 0, 100, {"status": "building features"})
        X_fc = make_lag_features(clean_scaled_filled, cfg.window_size, cov_scaled)
        y_fc = clean_scaled_filled[cfg.window_size:]
        target_valid = valid[cfg.window_size:]
        X_fc = X_fc[target_valid]
        y_fc = y_fc[target_valid]

        if progress_callback:
            progress_callback("forecast", 10, 100, {"status": "training"})

        self.forecast_model = HistGradientBoostingRegressor(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            validation_fraction=cfg.validation_fraction,
            early_stopping=True,
            random_state=cfg.random_state,
        )
        self.forecast_model.fit(X_fc, y_fc)
        self.history["forecast"] = {
            "loss": list(self.forecast_model.train_score_),
            "val_loss": list(self.forecast_model.validation_score_)
                if self.forecast_model.validation_score_ is not None else [],
            "n_iter": int(self.forecast_model.n_iter_),
        }

        if progress_callback:
            progress_callback("forecast", 100, 100,
                              {"status": f"done ({self.forecast_model.n_iter_} iter)"})

        # ---- Correction model: (raw lag window + cov) -> clean target ----
        if raw is not None:
            if progress_callback:
                progress_callback("correction", 0, 100, {"status": "building features"})
            raw_arr = raw.values.astype(float)
            raw_filled = np.where(
                np.isnan(raw_arr),
                self.scaler_target.mean_[0],
                raw_arr,
            )
            raw_scaled = self.scaler_target.transform(raw_filled.reshape(-1, 1)).ravel()

            X_co = make_lag_features(raw_scaled, cfg.window_size, cov_scaled)
            y_co = clean_scaled_filled[cfg.window_size:]
            target_valid_co = valid[cfg.window_size:]
            raw_valid = ~np.isnan(raw_arr)[cfg.window_size:]
            keep = target_valid_co & raw_valid
            X_co = X_co[keep]
            y_co = y_co[keep]

            if progress_callback:
                progress_callback("correction", 10, 100, {"status": "training"})

            self.correction_model = HistGradientBoostingRegressor(
                max_iter=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                validation_fraction=cfg.validation_fraction,
                early_stopping=True,
                random_state=cfg.random_state,
            )
            self.correction_model.fit(X_co, y_co)
            self.history["correction"] = {
                "loss": list(self.correction_model.train_score_),
                "val_loss": list(self.correction_model.validation_score_)
                    if self.correction_model.validation_score_ is not None else [],
                "n_iter": int(self.correction_model.n_iter_),
            }
            if progress_callback:
                progress_callback("correction", 100, 100,
                                  {"status": f"done ({self.correction_model.n_iter_} iter)"})

        return self.history

    # ---- Inference --------------------------------------------------------

    def forecast(
        self,
        series: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict per-timestep values using the forecast model.

        Returns a series aligned with input; first `window_size` samples are NaN.
        """
        if self.forecast_model is None:
            raise RuntimeError("Forecast model not trained.")
        return self._predict(series, covariates, self.forecast_model)

    def correct(
        self,
        raw: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict cleaned values from raw windows."""
        if self.correction_model is None:
            raise RuntimeError("Correction model not trained.")
        return self._predict(raw, covariates, self.correction_model)

    def _predict(self, series, covariates, model) -> pd.Series:
        cfg = self.config
        arr = series.values.astype(float)
        arr_filled = np.where(
            np.isnan(arr),
            self.scaler_target.mean_[0],
            arr,
        )
        arr_scaled = self.scaler_target.transform(arr_filled.reshape(-1, 1)).ravel()

        if self.covar_names and covariates is not None:
            cov = covariates[self.covar_names].values.astype(float)
            for j in range(cov.shape[1]):
                m = self.scaler_covar.mean_[j]
                cov[np.isnan(cov[:, j]), j] = m
            cov_scaled = self.scaler_covar.transform(cov)
        else:
            cov_scaled = None

        X = make_lag_features(arr_scaled, cfg.window_size, cov_scaled)
        if len(X) == 0:
            return pd.Series(np.nan, index=series.index)

        preds_scaled = model.predict(X)
        preds = self.scaler_target.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()

        out = np.full(len(arr), np.nan)
        out[cfg.window_size:cfg.window_size + len(preds)] = preds
        return pd.Series(out, index=series.index)

    # ---- Anomaly detection from residuals --------------------------------

    def detect_anomalies(
        self,
        raw: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Run forecast on raw stream, compute rolling threshold, flag points.

        Returns (anomaly_flags, residuals, threshold_series).
        """
        cfg = self.config
        preds = self.forecast(raw, covariates)
        residual = (raw - preds).abs()

        roll_mean = residual.rolling(cfg.rolling_threshold_window,
                                     min_periods=50).mean()
        roll_std = residual.rolling(cfg.rolling_threshold_window,
                                    min_periods=50).std()
        threshold = roll_mean + cfg.threshold_k * roll_std

        flags = (residual > threshold) & residual.notna()

        if cfg.widen_window > 0:
            flags = _widen_flags(flags, cfg.widen_window)

        return flags, residual, threshold

    # ---- Persistence ------------------------------------------------------

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.forecast_model is not None:
            with open(folder / "forecast.pkl", "wb") as f:
                pickle.dump(self.forecast_model, f)
        if self.correction_model is not None:
            with open(folder / "correction.pkl", "wb") as f:
                pickle.dump(self.correction_model, f)
        meta = {
            "config": asdict(self.config),
            "scaler_target": {
                "mean": self.scaler_target.mean_.tolist() if hasattr(self.scaler_target, "mean_") else None,
                "scale": self.scaler_target.scale_.tolist() if hasattr(self.scaler_target, "scale_") else None,
            },
            "scaler_covar": {
                "mean": self.scaler_covar.mean_.tolist() if hasattr(self.scaler_covar, "mean_") else None,
                "scale": self.scaler_covar.scale_.tolist() if hasattr(self.scaler_covar, "scale_") else None,
            },
            "covar_names": self.covar_names,
            "history": self.history,
            "model_kind": "sklearn_hgb",
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "ParameterLearned":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        obj = cls(LearnedConfig(**meta["config"]))
        # Restore scalers manually
        st = meta.get("scaler_target") or {}
        if st.get("mean") is not None:
            obj.scaler_target.mean_ = np.array(st["mean"])
            obj.scaler_target.scale_ = np.array(st["scale"])
            obj.scaler_target.var_ = obj.scaler_target.scale_ ** 2
            obj.scaler_target.n_features_in_ = len(obj.scaler_target.mean_)
        sc = meta.get("scaler_covar") or {}
        if sc.get("mean") is not None:
            obj.scaler_covar.mean_ = np.array(sc["mean"])
            obj.scaler_covar.scale_ = np.array(sc["scale"])
            obj.scaler_covar.var_ = obj.scaler_covar.scale_ ** 2
            obj.scaler_covar.n_features_in_ = len(obj.scaler_covar.mean_)
        obj.covar_names = meta["covar_names"]
        obj.history = meta.get("history", {})
        if (folder / "forecast.pkl").exists():
            with open(folder / "forecast.pkl", "rb") as f:
                obj.forecast_model = pickle.load(f)
        if (folder / "correction.pkl").exists():
            with open(folder / "correction.pkl", "rb") as f:
                obj.correction_model = pickle.load(f)
        return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _widen_flags(flags: pd.Series, window: int) -> pd.Series:
    """Dilate True regions by `window` samples on each side."""
    if not flags.any():
        return flags
    arr = flags.values.astype(int)
    kernel = np.ones(2 * window + 1, dtype=int)
    widened = np.convolve(arr, kernel, mode="same") > 0
    return pd.Series(widened, index=flags.index)


# ---------------------------------------------------------------------------
# Validation metrics  (identical to lstm_models.compute_metrics)
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: pd.Series,
    truth: pd.Series,
) -> dict:
    """Precision / recall / F1 from two boolean series."""
    pred = predictions.fillna(False).astype(bool)
    true = truth.fillna(False).astype(bool)
    common = pred.index.intersection(true.index)
    pred, true = pred.loc[common], true.loc[common]

    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }
