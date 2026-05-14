"""
LSTM models for water-quality anomaly detection and correction.
==============================================================
Implements PyHydroQC-style (Jones 2022) supervised workflow:

    Forecast model: trained on clean data, predicts next clean value from
                    past clean values + covariates. At inference, large
                    residuals on the raw stream = anomalies.

    Correction model: trained on (raw window, covariates) -> clean value
                      pairs, used to fill flagged gaps.

Anomaly detection uses a dynamic threshold:
    mean(|residual|) + k * std(|residual|)   over a rolling clean window
followed by Jones' "window widening" to catch full event extents.

Validation: precision / recall / F1 vs. derived ground truth (raw != clean
within tolerance) per Jones step 7.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# TensorFlow is heavy; import lazily so the rest of the package still works
# on installs without it.
_TF_AVAILABLE = None
def _tf():
    global _TF_AVAILABLE
    if _TF_AVAILABLE is None:
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
            _TF_AVAILABLE = tf
        except ImportError:
            _TF_AVAILABLE = False
    return _TF_AVAILABLE


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    """Per-parameter training settings."""
    parameter: str
    window_size: int = 96            # samples of history per prediction (24 hr @ 15 min)
    lstm_units: int = 64             # neurons per LSTM layer
    n_layers: int = 1
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    # Anomaly detection
    threshold_k: float = 4.0         # threshold = mean(|res|) + k*std(|res|)
    rolling_threshold_window: int = 1000  # samples used to compute threshold
    widen_window: int = 4            # widen flagged segments by N samples each side
    # Label derivation
    label_tolerance: float = 0.0     # |raw - clean| above this = anomaly

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "LSTMConfig":
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
    """Boolean label: True where raw differs from clean by more than tolerance.

    A NaN in raw or clean produces label = False (unknown, treated as clean).
    """
    diff = (raw - clean).abs()
    label = diff > tolerance
    label = label.where(raw.notna() & clean.notna(), False)
    return label


# ---------------------------------------------------------------------------
# Windowing utilities
# ---------------------------------------------------------------------------

def make_windows(
    series: np.ndarray,
    covariates: Optional[np.ndarray],
    window_size: int,
    targets: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Build (X, y) supervised windows.

    X has shape (n_windows, window_size, n_features) where n_features =
    1 (the parameter) + n_covariates.

    For forecast training, pass series=clean, targets=clean shifted by one.
    For correction training, pass series=raw, targets=clean[window_end].
    """
    n = len(series)
    series = series.reshape(-1, 1)
    if covariates is not None and covariates.size > 0:
        feats = np.concatenate([series, covariates], axis=1)
    else:
        feats = series

    n_windows = n - window_size
    if n_windows <= 0:
        return np.empty((0, window_size, feats.shape[1])), None

    X = np.lib.stride_tricks.sliding_window_view(
        feats, window_shape=window_size, axis=0
    )
    # sliding_window_view returns shape (n_windows, n_feats, window_size)
    X = X.transpose(0, 2, 1)[:n_windows]

    if targets is not None:
        y = targets[window_size:window_size + n_windows]
        return X, y
    return X, None


# ---------------------------------------------------------------------------
# Scalers - keep per-feature stats so we can invert
# ---------------------------------------------------------------------------

class Standardizer:
    """Robust z-score scaler: subtract median, divide by IQR/1.349."""
    def __init__(self):
        self.center_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        # X: 2D (n_samples, n_features) or 1D
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.center_ = np.nanmedian(X, axis=0)
        q25, q75 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q75 - q25
        # Fall back to std if IQR is 0
        scale = iqr / 1.349
        std = np.nanstd(X, axis=0)
        scale = np.where(scale > 0, scale, std)
        scale = np.where(scale > 0, scale, 1.0)
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        was_1d = X.ndim == 1
        if was_1d:
            X = X.reshape(-1, 1)
        out = (X - self.center_) / self.scale_
        return out.ravel() if was_1d else out

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        was_1d = X.ndim == 1
        if was_1d:
            X = X.reshape(-1, 1)
        out = X * self.scale_ + self.center_
        return out.ravel() if was_1d else out

    def to_dict(self) -> dict:
        return {
            "center": self.center_.tolist() if self.center_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        s = cls()
        s.center_ = np.array(d["center"]) if d["center"] is not None else None
        s.scale_ = np.array(d["scale"]) if d["scale"] is not None else None
        return s


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_lstm(
    window_size: int,
    n_features: int,
    units: int = 64,
    n_layers: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
):
    """Plain stacked LSTM with a dense head. Predicts a single scalar."""
    tf = _tf()
    if not tf:
        raise ImportError("TensorFlow is not installed.")
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential()
    model.add(Input(shape=(window_size, n_features)))
    for i in range(n_layers):
        is_last = i == n_layers - 1
        model.add(LSTM(units, return_sequences=not is_last))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


# ---------------------------------------------------------------------------
# Trainer: holds forecast + correction models for one parameter
# ---------------------------------------------------------------------------

class ParameterLSTM:
    """Forecast + correction LSTMs for a single parameter."""

    def __init__(self, config: LSTMConfig):
        self.config = config
        self.forecast_model = None
        self.correction_model = None
        self.scaler_target = Standardizer()
        self.scaler_covar = Standardizer()
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

        Returns a dict of training history (loss curves) per model.
        """
        tf = _tf()
        if not tf:
            raise ImportError("TensorFlow is required for LSTM training.")

        cfg = self.config

        # Fit scalers on clean data + covariates
        clean_arr = clean.values.astype(float)
        valid = ~np.isnan(clean_arr)
        self.scaler_target.fit(clean_arr[valid])

        if covariates is not None and len(covariates.columns) > 0:
            self.covar_names = list(covariates.columns)
            cov_arr = covariates.values.astype(float)
            # Fill NaN covariates with median for fitting
            for j in range(cov_arr.shape[1]):
                m = np.nanmedian(cov_arr[:, j])
                cov_arr[np.isnan(cov_arr[:, j]), j] = m
            self.scaler_covar.fit(cov_arr)
            cov_scaled = self.scaler_covar.transform(cov_arr)
        else:
            cov_scaled = None

        clean_scaled = self.scaler_target.transform(clean_arr)
        # Replace NaNs in clean target with 0 (will be masked when building windows)
        clean_scaled_filled = np.where(valid, clean_scaled, 0.0)

        # ---- Forecast model: predict next clean value from past clean + cov
        X_fc, y_fc = make_windows(
            clean_scaled_filled, cov_scaled, cfg.window_size,
            targets=clean_scaled_filled,
        )
        # Drop windows whose target was originally NaN
        target_valid = valid[cfg.window_size:cfg.window_size + len(y_fc)]
        X_fc = X_fc[target_valid]
        y_fc = y_fc[target_valid]

        n_feats = X_fc.shape[2]
        self.forecast_model = build_lstm(
            cfg.window_size, n_feats,
            units=cfg.lstm_units, n_layers=cfg.n_layers,
            dropout=cfg.dropout, learning_rate=cfg.learning_rate,
        )
        cb = []
        if progress_callback:
            cb.append(_CallbackForward(progress_callback, "forecast", cfg.epochs))

        hist_fc = self.forecast_model.fit(
            X_fc, y_fc,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            validation_split=cfg.validation_split,
            verbose=0, callbacks=cb,
        )
        self.history["forecast"] = {
            "loss": hist_fc.history["loss"],
            "val_loss": hist_fc.history.get("val_loss", []),
        }

        # ---- Correction model: (raw window + cov) -> clean target
        if raw is not None:
            raw_arr = raw.values.astype(float)
            # Scale raw with the same scaler as clean
            raw_scaled = self.scaler_target.transform(
                np.where(np.isnan(raw_arr), self.scaler_target.center_, raw_arr)
            )
            X_co, y_co = make_windows(
                raw_scaled, cov_scaled, cfg.window_size,
                targets=clean_scaled_filled,
            )
            target_valid_co = valid[cfg.window_size:cfg.window_size + len(y_co)]
            raw_valid = ~np.isnan(raw_arr)[cfg.window_size:cfg.window_size + len(y_co)]
            keep = target_valid_co & raw_valid
            X_co = X_co[keep]
            y_co = y_co[keep]

            self.correction_model = build_lstm(
                cfg.window_size, n_feats,
                units=cfg.lstm_units, n_layers=cfg.n_layers,
                dropout=cfg.dropout, learning_rate=cfg.learning_rate,
            )
            cb2 = []
            if progress_callback:
                cb2.append(_CallbackForward(progress_callback, "correction", cfg.epochs))

            hist_co = self.correction_model.fit(
                X_co, y_co,
                epochs=cfg.epochs, batch_size=cfg.batch_size,
                validation_split=cfg.validation_split,
                verbose=0, callbacks=cb2,
            )
            self.history["correction"] = {
                "loss": hist_co.history["loss"],
                "val_loss": hist_co.history.get("val_loss", []),
            }

        return self.history

    # ---- Inference --------------------------------------------------------

    def forecast(
        self,
        series: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict per-timestep values using the forecast LSTM.

        Returns a series aligned with the input; the first `window_size`
        samples are NaN (no history available to predict).
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
        # Fill NaN with median for windowing (these spots will be flagged anyway)
        arr_filled = np.where(np.isnan(arr), self.scaler_target.center_, arr)
        arr_scaled = self.scaler_target.transform(arr_filled)

        if self.covar_names and covariates is not None:
            cov = covariates[self.covar_names].values.astype(float)
            for j in range(cov.shape[1]):
                m = self.scaler_covar.center_[j]
                cov[np.isnan(cov[:, j]), j] = m
            cov_scaled = self.scaler_covar.transform(cov)
        else:
            cov_scaled = None

        X, _ = make_windows(arr_scaled, cov_scaled, cfg.window_size)
        if len(X) == 0:
            return pd.Series(np.nan, index=series.index)

        preds_scaled = model.predict(X, verbose=0).ravel()
        preds = self.scaler_target.inverse_transform(preds_scaled)

        # Align: predictions start at index window_size
        out = np.full(len(arr), np.nan)
        out[cfg.window_size:cfg.window_size + len(preds)] = preds
        return pd.Series(out, index=series.index)

    # ---- Anomaly detection from residuals ---------------------------------

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

        # Dynamic threshold: rolling mean + k * rolling std of |residual|
        roll_mean = residual.rolling(cfg.rolling_threshold_window,
                                     min_periods=50).mean()
        roll_std = residual.rolling(cfg.rolling_threshold_window,
                                    min_periods=50).std()
        threshold = roll_mean + cfg.threshold_k * roll_std

        flags = (residual > threshold) & residual.notna()

        # Jones step 6: widen flagged segments
        if cfg.widen_window > 0:
            flags = _widen_flags(flags, cfg.widen_window)

        return flags, residual, threshold

    # ---- Persistence ------------------------------------------------------

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.forecast_model is not None:
            self.forecast_model.save(folder / "forecast.keras")
        if self.correction_model is not None:
            self.correction_model.save(folder / "correction.keras")
        meta = {
            "config": asdict(self.config),
            "scaler_target": self.scaler_target.to_dict(),
            "scaler_covar": self.scaler_covar.to_dict(),
            "covar_names": self.covar_names,
            "history": self.history,
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "ParameterLSTM":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        obj = cls(LSTMConfig(**meta["config"]))
        obj.scaler_target = Standardizer.from_dict(meta["scaler_target"])
        obj.scaler_covar = Standardizer.from_dict(meta["scaler_covar"])
        obj.covar_names = meta["covar_names"]
        obj.history = meta.get("history", {})
        tf = _tf()
        if (folder / "forecast.keras").exists() and tf:
            obj.forecast_model = tf.keras.models.load_model(folder / "forecast.keras")
        if (folder / "correction.keras").exists() and tf:
            obj.correction_model = tf.keras.models.load_model(folder / "correction.keras")
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


class _CallbackForward:
    """Adapter to pipe Keras epoch progress to a callable (e.g. Streamlit)."""
    def __init__(self, callback, phase, total_epochs):
        self.callback = callback
        self.phase = phase
        self.total = total_epochs
        # Keras Callback API
        try:
            from tensorflow.keras.callbacks import Callback
            base = Callback
        except Exception:
            base = object
        # Build a proper Callback subclass dynamically
        self._cb = self._make_callback(base)

    def _make_callback(self, base):
        outer = self
        class _Cb(base):
            def on_epoch_end(self, epoch, logs=None):
                outer.callback(outer.phase, epoch + 1, outer.total, logs or {})
        return _Cb()

    # Allow this object to be used directly in callbacks=[...]
    def __getattr__(self, name):
        return getattr(self._cb, name)


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: pd.Series,
    truth: pd.Series,
) -> dict:
    """Precision / recall / F1 from two boolean series."""
    pred = predictions.fillna(False).astype(bool)
    true = truth.fillna(False).astype(bool)
    # Align
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
