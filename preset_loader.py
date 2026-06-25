"""
Station preset loader.
======================
Discovers preset JSON files in a folder and converts them into the
ParameterConfig objects the QC engine uses. Also provides column-name
auto-mapping based on each preset's `column_aliases`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from water_quality_qc_v2 import ParameterConfig


def list_presets(folder: str | Path = "presets") -> list[dict]:
    """Scan the preset folder and return summaries of every valid preset.

    Each summary: {file, station_id, station_name, description, n_parameters}
    Files starting with '_' are sorted last (so generic templates appear
    after station-specific presets).
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    out = []
    for f in sorted(folder.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            out.append({
                "file": str(f),
                "filename": f.name,
                "station_id": data.get("station_id", f.stem),
                "station_name": data.get("station_name", f.stem),
                "description": data.get("description", ""),
                "n_parameters": len(data.get("parameters", {})),
                "_data": data,
            })
        except Exception as e:
            out.append({
                "file": str(f),
                "filename": f.name,
                "station_id": f.stem,
                "station_name": f"⚠️ {f.name} (invalid: {e})",
                "description": "",
                "n_parameters": 0,
                "_data": None,
            })

    # Stable sort: templates (prefix '_') to the bottom
    out.sort(key=lambda p: (p["filename"].startswith("_"), p["filename"].lower()))
    return out


def load_preset(path: str | Path) -> dict:
    """Load and return a single preset's raw JSON dict."""
    return json.loads(Path(path).read_text())


def configs_from_preset(preset: dict) -> dict[str, ParameterConfig]:
    """Convert a preset's parameter dict to a {name: ParameterConfig} map."""
    out = {}
    for name, settings in preset.get("parameters", {}).items():
        # Whitelist keys that map to ParameterConfig fields; ignore extras
        valid_keys = set(ParameterConfig.__dataclass_fields__.keys())
        clean_settings = {
            "name": name,
            **{k: v for k, v in settings.items() if k in valid_keys},
        }
        out[name] = ParameterConfig(**clean_settings)
    return out


def auto_map_columns(
    df_columns: list[str],
    preset: dict,
) -> dict[str, Optional[str]]:
    """Given a dataframe's column list and a preset, suggest column mappings.

    Returns a dict like:
        {
          "timestamp": "Timestamp (UTC-05:00)",
          "pH": "pH",
          "temperature": "Water Temp",
          "rainfall": "Precip",   # or None if no match
          ...
        }

    Case-insensitive, partial-match-friendly:
      - Exact match (case-insensitive) wins.
      - Otherwise, first preset alias that's a substring of a column name
        (or vice versa) wins.
    """
    aliases = preset.get("column_aliases", {})
    cols_lower = {c.lower(): c for c in df_columns}

    mapping: dict[str, Optional[str]] = {}
    for canonical, alias_list in aliases.items():
        match = None
        # Pass 1: exact case-insensitive
        for alias in alias_list:
            if alias.lower() in cols_lower:
                match = cols_lower[alias.lower()]
                break
        # Pass 2: substring
        if match is None:
            for alias in alias_list:
                al = alias.lower()
                for c_lower, c_orig in cols_lower.items():
                    if al in c_lower or c_lower in al:
                        match = c_orig
                        break
                if match:
                    break
        mapping[canonical] = match

    return mapping


def preset_to_session_state(preset: dict) -> dict:
    """Build a dict of UI-state defaults from a preset.

    Used by the Streamlit app to pre-fill widgets when a preset is selected.
    """
    cov = preset.get("covariates", {})
    return {
        "preset_id": preset.get("station_id"),
        "configs": configs_from_preset(preset),
        "rain_window_hr": float(cov.get("rain_window_hr", 1.0)),
        "rain_event_threshold": float(cov.get("rain_event_threshold", 0.05)),
        "stage_high_quantile": float(cov.get("stage_high_quantile", 0.90)),
        "rainfall_col_hint": cov.get("rainfall_col_hint"),
        "stage_col_hint": cov.get("stage_col_hint"),
    }
