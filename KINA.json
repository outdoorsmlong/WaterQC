{
  "station_id": "KINA",
  "station_name": "KIN_A — Cooper River tidal station",
  "description": "87 days of 15-min data analyzed (Feb–May 2026). pH shows diurnal cycles; turbidity is bursty during storms; Spec Cond can flatline for days during stable conditions. Watch for -99999 sentinel values from the logger.",
  "sampling_interval_minutes": 15,
  "column_aliases": {
    "timestamp": ["Timestamp (UTC-05:00)", "TimeStamp", "datetime", "time"],
    "pH": ["pH", "pH@COLA_KINA"],
    "temperature": ["Water Temp", "Water Temp@COLA_KINA", "Temperature"],
    "turbidity": ["Turb", "Turbidity"],
    "specific_conductivity": ["Spec Cond", "SpC"],
    "dissolved_oxygen": ["O2", "O2 mg/L", "DO"],
    "stage": ["Stage"],
    "rainfall": ["Precip", "Precipitation", "Rainfall"]
  },
  "covariates": {
    "rainfall_col_hint": "Precip",
    "stage_col_hint": "Stage",
    "rain_window_hr": 2.0,
    "rain_event_threshold": 0.02,
    "stage_high_quantile": 0.90
  },
  "parameters": {
    "temperature": {
      "units": "°F",
      "range_min": 32,
      "range_max": 90,
      "spike_threshold": 5.0,
      "persistence_window": 12,
      "max_rate_change": 6.0,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": false,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": false
    },
    "specific_conductivity": {
      "units": "mS/cm",
      "range_min": 0.0,
      "range_max": 0.5,
      "spike_threshold": 4.0,
      "persistence_window": 384,
      "max_rate_change": 0.2,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": true,
      "notes": "Persistence window set to 384 samples (96 hr / 4 days) because this sensor frequently flatlines for 1-3 days at a time. Shorter windows flag 30-60% of the dataset. Investigate sensor maintenance separately. Watch for -99999 logger sentinel — range_min catches it."
    },
    "pH": {
      "units": "pH",
      "range_min": 6.0,
      "range_max": 9.0,
      "spike_threshold": 4.0,
      "persistence_window": 12,
      "max_rate_change": 0.6,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": true,
      "notes": "Technician corrections smooth diurnal cycles. Rules-based engine preserves them."
    },
    "dissolved_oxygen": {
      "units": "mg/L",
      "range_min": 0.0,
      "range_max": 15.0,
      "spike_threshold": 5.0,
      "persistence_window": 12,
      "max_rate_change": 7.0,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": true,
      "use_covariates_for_correction": true
    },
    "turbidity": {
      "units": "FNU",
      "range_min": 0.0,
      "range_max": 1000.0,
      "spike_threshold": 6.0,
      "persistence_window": 12,
      "max_rate_change": 300.0,
      "suppress_spike_in_rain": true,
      "suppress_rate_in_rain": true,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": true,
      "notes": "High max-rate and spike-k because real storm pulses reach 600+ FNU."
    },
    "stage": {
      "units": "ft",
      "range_min": 0.5,
      "range_max": 8.0,
      "spike_threshold": 4.5,
      "persistence_window": 12,
      "max_rate_change": 0.7,
      "suppress_spike_in_rain": false,
      "suppress_rate_in_rain": false,
      "suppress_range_min_in_high_stage": false,
      "use_covariates_for_correction": false
    }
  },
  "source": "Derived from KIN_A_RAW_TEST.csv + KINA_Test_Time_Series_Data corrected dataset",
  "version": "1.0",
  "last_updated": "2026-05-14"
}
