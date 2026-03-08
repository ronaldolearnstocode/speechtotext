"""Load configuration from config.yaml with defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

_DEFAULTS = {
    "hotkey": "ctrl+win",
    "quit_hotkey": "ctrl+shift+q",
    "model_name": "large-v3",
    "device": "cpu",
    "compute_type": "int8",
    "sample_rate": 16000,
    "chunk_duration_ms": 30,
    "vad_aggressiveness": 2,
    "type_interval": 0.02,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML file, with env override and defaults."""
    config = dict(_DEFAULTS)
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(path)
    if path.exists() and yaml is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if v is not None:
                    config[k] = v
    # Env overrides (optional)
    if os.environ.get("STT_MODEL"):
        config["model_name"] = os.environ["STT_MODEL"]
    if os.environ.get("STT_DEVICE"):
        config["device"] = os.environ["STT_DEVICE"]
    return config
