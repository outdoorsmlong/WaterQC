# Station Preset Library

Each `.json` file in this folder is a **station preset** — a saved configuration
for one monitoring station that the app loads with a single dropdown click.

## How presets work

The app scans this folder on startup and lists every `.json` file in a
dropdown at the top of the Rules-based QC tab. Selecting a preset auto-fills:

- The expected column names (rainfall, stage, parameter columns)
- Per-parameter thresholds (range, spike, persistence, max rate)
- Event-aware behavior (which flags to suppress during rain/high stage)
- Covariate settings (rolling rainfall window, stage quantile)

Any auto-detected column that matches an alias in the preset is mapped
automatically. You can override any value in the UI after loading — presets
are starting points, not locked configs.

## Built-in presets

| File | Use for |
|---|---|
| `KINA.json` | KIN_A station — Cooper River-area site, derived from 87 days of corrected data |
| `_generic_freshwater.json` | Default starting point for any new freshwater stream |
| `_generic_tidal.json` | Estuarine/brackish sites with high specific conductivity |
| `_generic_mountain.json` | Cold-water oligotrophic mountain streams |

Files starting with `_` are sorted to the bottom of the dropdown so your
station-specific presets show first.

## Adding a new station

1. Copy an existing preset that resembles your site (e.g., `KINA.json`).
2. Rename it to your station ID (e.g., `KIN_B.json`, `Site_42.json`).
3. Open in any text editor and update:
   - `station_id`, `station_name`, `description`
   - `column_aliases` — list every name your CSVs use for each parameter
   - `parameters.<param>.range_min`, `range_max`, etc.
4. Save. The next time the app starts, your new preset appears in the dropdown.

## Deriving thresholds from your data

For each parameter, run this against ~3 months of your cleanest data:

| Threshold | How to derive |
|---|---|
| `range_min` / `range_max` | Round outside the observed [p0.1, p99.9] to nice numbers |
| `spike_threshold` | Start at 4.0; raise if too many false positives, lower if missing real spikes |
| `persistence_window` | Sampling-rate dependent: 12 samples × 15 min = 3 hours |
| `max_rate_change` | p99.9 of `|series.diff()|`, multiplied by samples-per-hour, rounded up |

The KINA preset is a worked example — it was derived from real raw + clean
data using exactly this approach.

## Schema reference

```json
{
  "station_id": "string, short identifier",
  "station_name": "string, human-readable name",
  "description": "string, free-form notes",
  "sampling_interval_minutes": 15,
  "column_aliases": {
    "timestamp":    ["list", "of", "possible", "column", "names"],
    "pH":           ["..."],
    "temperature":  ["..."],
    "turbidity":    ["..."],
    "specific_conductivity": ["..."],
    "dissolved_oxygen": ["..."],
    "stage":        ["..."],
    "rainfall":     ["..."]
  },
  "covariates": {
    "rainfall_col_hint":  "preferred column name if available",
    "stage_col_hint":     "preferred column name if available",
    "rain_window_hr":     1.0,
    "rain_event_threshold": 0.05,
    "stage_high_quantile":  0.90
  },
  "parameters": {
    "temperature": {
      "units":              "°F",
      "range_min":          32,
      "range_max":          95,
      "spike_threshold":    4.0,
      "persistence_window": 12,
      "max_rate_change":    9.0,
      "suppress_spike_in_rain":            false,
      "suppress_rate_in_rain":             false,
      "suppress_range_min_in_high_stage":  false,
      "use_covariates_for_correction":     false,
      "notes":              "optional free-form string"
    }
  },
  "version":      "1.0",
  "last_updated": "YYYY-MM-DD"
}
```
