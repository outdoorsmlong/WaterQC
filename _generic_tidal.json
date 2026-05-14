{
  "station_id": "GENERIC_TIDAL",
  "station_name": "Generic tidal/estuarine site (template)",
  "description": "Settings for brackish/estuarine sites where specific conductivity is much higher than freshwater and varies with tidal cycles. Adjust ranges based on your specific salinity regime.",
  "sampling_interval_minutes": 15,
  "column_aliases": {
    "timestamp": ["datetime", "TimeStamp", "Timestamp", "time"],
    "pH": ["pH", "ph", "PH"],
    "temperature": ["temperature", "Water Temp", "Temp"],
    "turbidity": ["turbidity", "Turb", "Turbidity"],
    "specific_conductivity": ["specific_conductivity", "Spec Cond", "SpC", "Salinity"],
    "dissolved_oxygen": ["dissolved_oxygen", "DO", "O2"],
    "stage": ["Stage", "TideHeight", "WaterLevel"],
    "rainfall": ["rainfall", "Precip", "Rain"]
  },
  "covariates": {
    "rainfall_col_hint": "rainfall",
    "stage_col_hint": "stage",
    "rain_window_hr": 1.0,
    "rain_event_threshold": 0.05,
    "stage_high_quantile": 0.90
  },
  "parameters": {
    "temperature": {
      "units": "°F",
      "range_min": 32,
      "range_max": 95,
      "spike_threshold": 4.0,
      "persistence_window": 12,
      "max_rate_change": 6.0,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": false,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": false
    },
    "specific_conductivity": {
      "units": "mS/cm",
      "range_min": 0.1,
      "range_max": 60,
      "spike_threshold": 4.0,
      "persistence_window": 12,
      "max_rate_change": 10,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": true,
      "notes": "Range covers freshwater (~0.5) through seawater (~50). Tidal cycles can swing SpC by 10s of mS/cm per hour — max rate is generous."
    },
    "pH": {
      "units": "pH",
      "range_min": 6.5,
      "range_max": 9.0,
      "spike_threshold": 3.5,
      "persistence_window": 12,
      "max_rate_change": 0.8,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": true
    },
    "dissolved_oxygen": {
      "units": "mg/L",
      "range_min": 0,
      "range_max": 16,
      "spike_threshold": 4.0,
      "persistence_window": 12,
      "max_rate_change": 5.0,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": true,
      "use_covariates_for_correction": true,
      "notes": "Estuarine systems often have real hypoxia events; suppress low-DO flags during high stage."
    },
    "turbidity": {
      "units": "NTU",
      "range_min": 0,
      "range_max": 1500,
      "spike_threshold": 5.5,
      "persistence_window": 12,
      "max_rate_change": 100,
      "suppress_spike_in_rain": true,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": true
    },
    "stage": {
      "units": "ft",
      "range_min": -2,
      "range_max": 15,
      "spike_threshold": 4.0,
      "persistence_window": 12,
      "max_rate_change": 3.0,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": false,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": false,
      "notes": "Negative stage minimum allows for low-tide values below local datum."
    }
  },
  "version": "1.0",
  "last_updated": "2026-05-14"
}
