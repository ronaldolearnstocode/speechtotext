"""Load configuration from config.yaml with defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from speechtotext.paths import get_app_root, get_bundle_root

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
    "vad_filter": True,
    "vad_min_silence_duration_ms": 500,
    "vad_filter_capture": False,
    "type_interval": 0.02,
    "show_window_on_start": False,
    "assistant_enabled": True,
    "assistant_wake_word_map": {"jarvis": "ollama", "mother": "gemini"},
    "assistant_provider_enabled": {"ollama": True, "gemini": False},
    "assistant_mode": "work",
    "assistant_model_work": "qwen2.5-coder:7b",
    "assistant_model_quality": "qwen2.5-coder:14b",
    "assistant_ollama_host": "http://127.0.0.1:11434",
    "assistant_timeout_s": 60,
    "assistant_temperature": 0.3,
    "assistant_max_tokens": 256,
    "assistant_tts_provider": "windows",
    "assistant_tts_voice": "male",
    "assistant_tts_rate": 0,
    "assistant_tts_volume": 100,
    "assistant_tts_piper_path": "",
    "assistant_tts_piper_model": "",
    "assistant_output_window_enabled": True,
    "assistant_output_window_topmost": False,
    "assistant_voice_summary_only": True,
    "assistant_voice_summary_max_chars": 160,
    "assistant_partial_wake_enabled": True,
    "assistant_partial_wake_first_window_ms": 1000,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML file, with env override and defaults."""
    config = dict(_DEFAULTS)
    if path is None:
        path = get_app_root() / "config.yaml"
        if not path.exists():
            path = get_bundle_root() / "config.yaml"
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
