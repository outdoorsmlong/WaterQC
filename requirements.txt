# 💧 Water Quality QC Tool

A web-based UI for the water quality QC framework. The Python QC engine runs
in the background — you interact through a browser.

## What you got

| File / folder | Purpose |
|------|---------|
| `water_quality_qc_app.py` | The web UI (Streamlit) |
| `water_quality_qc_v2.py` | The rules-based QC engine |
| `lstm_models.py` | LSTM forecast + correction models (PyHydroQC-style) |
| `preset_loader.py` | Loads station presets from JSON |
| `presets/` | **Station preset library** — one JSON per monitoring station |
| `requirements.txt` | Libraries to install |
| `Run_QC_Tool.bat` | Windows launcher (double-click) |
| `Run_QC_Tool.sh` | Mac/Linux launcher |

## Station presets

The **`presets/`** folder holds one JSON file per monitoring station, each
capturing column-name aliases, parameter thresholds, and event-aware
behavior. When you open the app, a dropdown at the top of the Rules-based
QC tab lets you pick a preset — column mappings and thresholds fill in
automatically. You can override any value before running.

Built-in presets:

- **KINA** — Cooper River-area site, thresholds derived from 87 days of real corrected data
- **SMIB** — Smith Branch (Columbia, SC), thresholds derived from a full year (2021) of raw + corrected data
- **Generic Campbell** — template for Campbell PakBus logger exports with mixed cadences
- **Generic freshwater** — conservative defaults for new freshwater sites
- **Generic tidal** — for brackish/estuarine sites
- **Generic mountain** — for cold-water oligotrophic streams

**Adding your own station:** copy any existing JSON in `presets/`, rename
it (e.g., `MyStation.json`), edit the parameter thresholds and column
aliases for your site, save. The next time you open the app, your new
preset appears in the dropdown. Full schema documented in `presets/README.md`.

## Learned-model workflow (Jones 2022 / PyHydroQC-style)

The **Train Model** and **Detect & Validate** tabs implement the supervised
workflow from Amber Jones' PyHydroQC paper, using **scikit-learn gradient
boosting** by default (lightweight, installs anywhere, trains in seconds).
Optionally, you can switch to a **TensorFlow LSTM** backend by setting the
`WQC_USE_LSTM=1` environment variable, if you have TF installed and want
the extra modeling capacity.

Workflow:

1. **Retrieve data** — raw uncorrected stream loaded on the main page.
2. **Rules-based screening** — the *Rules-based QC* tab handles this.
3. **Develop a model** — *Train Model* tab. Upload your *clean* (corrected)
   dataset with matching timestamps. The app trains two models per parameter:
   a **forecast** model (predicts the next clean value from past + covariates)
   and a **correction** model (maps raw windows directly to clean values).
4. **Apply model** — *Detect & Validate* tab. Runs the forecast model on
   the raw stream and computes residuals.
5. **Dynamic threshold** — anomalies are flagged where `|residual| >
   mean(|residual|) + k * std(|residual|)` over a rolling window. k is
   tunable (default 4.0).
6. **Window widening** — flagged points get their neighbors checked too,
   so you catch the full extent of each event rather than just the peak.
7. **Validation metrics** — precision, recall, F1, and confusion-matrix
   counts vs. ground truth derived from your clean dataset (`raw ≠ clean`
   within a per-parameter tolerance).
8. **Correction** — flagged points can be replaced with the model's output,
   or fall back to the rules-based regression approach.

Trained models are saved to `./models/<parameter>/` so training is a
one-time cost. Future runs just load and apply them.

### Why sklearn instead of TensorFlow by default?

TensorFlow doesn't yet ship wheels for Python 3.13/3.14, has frequent
Windows install pain (~500 MB footprint, DLL issues), and often OOMs on
Streamlit Cloud's free tier. Scikit-learn installs cleanly on every Python
version, trains in seconds on CPU, and produces models small enough to
commit to a repo (~1 MB each).

For most water-quality QC tasks the predictive-quality gap is small. Tested
on synthetic pH data with diurnal cycles and three anomaly types, the
sklearn backend matched the LSTM on F1 and produced *more accurate*
corrections (mean error 0.017 pH vs 0.063 for LSTM). If you have specific
patterns where the LSTM is clearly better, you can switch backends at
runtime via `WQC_USE_LSTM=1`.

## Setup (one time)

1. **Install Python 3.9 or newer** from https://www.python.org/downloads/
   - On Windows, check **"Add Python to PATH"** during install.
2. **Install libraries.** Open Terminal / Command Prompt in this folder:
   ```
   pip install -r requirements.txt
   ```
   That's it — scikit-learn is in the main requirements. No separate ML
   install needed.

   Optional — only if you specifically want TensorFlow LSTM models (Python
   3.10–3.12 only):
   ```
   pip install -r requirements-lstm.txt
   ```
   Then set `WQC_USE_LSTM=1` in your environment before launching the app.

## Deploying to Streamlit Cloud

**Important:** Streamlit Cloud picks the Python version at deploy time via
the **Advanced settings** dialog. Files like `.python-version` and
`runtime.txt` are **ignored** ([Streamlit
docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)).
To change Python version after deployment, you must **delete the app and
redeploy**.

**For rules-based features only (recommended for cloud):**
- Any supported Python version works (3.10, 3.11, 3.12, 3.13, 3.14)
- Just deploy with the default `requirements.txt` — no extra steps

**For LSTM features on the cloud (advanced):**
1. Delete the app, then redeploy. In **Advanced settings** at deploy time,
   select Python 3.11 or 3.12 from the dropdown (TensorFlow doesn't support
   3.13+ yet).
2. Add these lines to `requirements.txt` (Streamlit Cloud only reads this
   one file):
   ```
   tensorflow>=2.15,<2.20
   scikit-learn>=1.3
   ```
3. Push and reboot.
4. Note: training large models on Streamlit Cloud free tier may OOM.
   Consider training locally and committing the `models/` folder instead.

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

## File format support

**CSV** with auto-detection of:
- Encoding (UTF-8, UTF-8-BOM, UTF-16, CP1252, Latin-1)
- Delimiter (comma, tab, semicolon)
- Header row position — multi-line metadata headers from AQUARIUS Time-Series
  and Campbell PakBus loggers are auto-skipped
- Mixed-cadence columns (e.g. 5-min rain + 15-min sonde in one file) —
  the app detects and offers to **auto-resample** to a uniform grid,
  using `sum` for rainfall (so totals are preserved exactly) and
  nearest-neighbor for everything else

**Excel (`.xlsx`, `.xls`)** — including AQUARIUS-style exports with
parameters laid out side-by-side, each with its own timestamp column.

**Logger sentinel handling**: each preset's `range_min` and `range_max`
catch logger error values (e.g. KINA's `-99999` for sonde dropouts) without
needing per-station custom code.

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
