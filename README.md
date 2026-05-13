# 💧 Water Quality QC Tool

A web-based UI for the water quality QC framework. The Python QC engine runs
in the background — you interact through a browser.

## What you got

| File | Purpose |
|------|---------|
| `water_quality_qc_app.py` | The web UI (Streamlit) |
| `water_quality_qc_v2.py` | The QC engine (called by the UI) |
| `requirements.txt` | Libraries to install |
| `Run_QC_Tool.bat` | Windows launcher (double-click) |
| `Run_QC_Tool.sh` | Mac/Linux launcher |

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

Just needs a timestamp column and one or more numeric parameter columns:

```
datetime,temperature,ph,dissolved_oxygen,turbidity,rainfall,stage
2024-01-01 00:00,58.2,7.51,9.1,4.8,0.0,2.51
2024-01-01 00:15,58.4,7.49,9.0,4.9,0.0,2.52
...
```

**Optional covariates** — rainfall and stage columns let the tool:
1. **Suppress false-positive flags during real hydrologic events.** A turbidity
   spike during a rain event is real, not a sensor fault. So is a DO drop
   during a flood, or a conductivity dilution during a storm.
2. **Inform correction estimates via regression.** When a point IS flagged
   as a fault, the corrected value comes from a regression that uses
   rainfall and stage as predictors (cross-faded with linear interpolation
   at the gap edges, PyHydroQC-style).

Column names are flexible — you map them in the UI. Temperature defaults
are in **°F**; the DO saturation check converts to °C internally.

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
