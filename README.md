# 💧 Water Quality QC Tool

A web-based UI for the water quality QC framework. The Python QC engine runs
in the background — you interact through a browser.

## What you got

| File | Purpose |
|------|---------|
| `water_quality_qc_app.py` | The web UI (Streamlit) |
| `water_quality_qc_v2.py` | The rules-based QC engine |
| `lstm_models.py` | LSTM forecast + correction models (PyHydroQC-style) |
| `requirements.txt` | Libraries to install |
| `Run_QC_Tool.bat` | Windows launcher (double-click) |
| `Run_QC_Tool.sh` | Mac/Linux launcher |

## LSTM workflow (Jones 2022 / PyHydroQC)

The **Train LSTM** and **LSTM Detect & Validate** tabs implement the supervised
workflow from Amber Jones' PyHydroQC paper:

1. **Retrieve data** — raw uncorrected stream loaded on the main page.
2. **Rules-based screening** — the *Rules-based QC* tab handles this.
3. **Develop a model** — *Train LSTM* tab. Upload your *clean* (corrected)
   dataset with matching timestamps. The app trains two models per parameter:
   a **forecast** LSTM (predicts the next clean value from past + covariates)
   and a **correction** LSTM (maps raw windows directly to clean values).
4. **Apply model** — *LSTM Detect & Validate* tab. Runs the forecast model
   on the raw stream and computes residuals.
5. **Dynamic threshold** — anomalies are flagged where `|residual| >
   mean(|residual|) + k * std(|residual|)` over a rolling window. k is
   tunable (default 4.0).
6. **Window widening** — flagged points get their neighbors checked too,
   so you catch the full extent of each event rather than just the peak.
7. **Validation metrics** — precision, recall, F1, and confusion-matrix
   counts vs. ground truth derived from your clean dataset (`raw ≠ clean`
   within a per-parameter tolerance).
8. **Correction** — flagged points can be replaced with the correction
   LSTM's output, or fall back to the rules-based regression approach.

Trained models are saved as `.keras` files in `./models/<parameter>/` so
training is a one-time cost. Future runs just load the model and apply it.

## Setup (one time)

1. **Install Python 3.9+** from https://www.python.org/downloads/
   - ⚠️ On Windows, check **"Add Python to PATH"** during install.
2. **Install libraries.** Open Terminal / Command Prompt in this folder:
   ```
   pip install -r requirements.txt
   ```
   (Takes 2–3 minutes.)

## Run it

- **Windows:** double-click `Run_QC_Tool.bat`
- **Mac/Linux:** double-click `Run_QC_Tool.sh` (or `bash Run_QC_Tool.sh` in Terminal)
- **Manual:** `streamlit run water_quality_qc_app.py`

Your browser opens at `http://localhost:8501`. Press **Ctrl+C** in the terminal to stop.

## Using the app

1. **Sidebar → Data source.** Upload a CSV or click *Generate demo data*.
2. **Map columns.** Pick your timestamp column and which parameter columns to QC.
3. **Configure thresholds.** Expand each parameter to tweak ranges, spike sensitivity,
   persistence window, max rate of change, and ARIMA on/off. Defaults are sensible.
4. **Run QC.** Click the button. Results appear below.
5. **Download** the flagged CSV, summary report, or both as a zip.

## CSV format

**Water quality file** — timestamp column + one or more numeric parameter columns:

```
datetime,temperature,ph,dissolved_oxygen,turbidity
2024-01-01 00:00,58.2,7.51,9.1,4.8
2024-01-01 00:15,58.4,7.49,9.0,4.9
...
```

**Rainfall and stage files** — separate uploads, with their own timestamp columns:

```
# rainfall.csv (might be 5-min data from a different logger)
TIMESTAMP,rain_inches
2024-01-01 00:00:00,0.000
2024-01-01 00:05:00,0.001
...

# stage.csv (might be 1-min data from a PT)
date_time,stage_ft
2024-01-01 00:00:00,2.51
2024-01-01 00:01:00,2.51
...
```

**You don't need matching timestamps.** The app detects each file's timestamp
column automatically and uses nearest-neighbor matching (within your chosen
tolerance, default 10 min) to align everything onto the WQ grid. The
alignment log shows you the match rate, so you can spot timezone mismatches
or logger drift.

**Optional covariates** let the tool:
1. **Suppress false-positive flags during real hydrologic events.** A turbidity
   spike during a rain event is real, not a sensor fault. So is a DO drop
   during a flood, or a conductivity dilution during a storm.
2. **Inform correction estimates via regression.** When a point IS flagged
   as a fault, the corrected value comes from a regression that uses
   rainfall and stage as predictors (cross-faded with linear interpolation
   at the gap edges, PyHydroQC-style).

Column names are flexible — the app maps them automatically. Temperature
defaults are in **°F**; the DO saturation check converts to °C internally.

## Customizing the engine

The UI is the easy entry point, but `water_quality_qc_v2.py` is plain Python.
You can use it directly from a script or notebook:

```python
import pandas as pd
from water_quality_qc_v2 import WaterQualityQCv2

df = pd.read_csv("your_data.csv")
qc = WaterQualityQCv2(
    data=df,
    timestamp_col="datetime",
    parameters=["temperature", "ph", "dissolved_oxygen", "turbidity"],
)
qc.run_all_sequential()
qc.export_flagged_csv("flagged.csv")
qc.write_summary_report("summary.txt")
```

That's the adaptability: UI for everyday use, Python underneath when you want it.
