"""Resolve app root for config and data files (works when frozen with PyInstaller)."""

from __future__ import annotations

import sys
from pathlib import Path


def get_app_root() -> Path:
    """Return the app root directory.

    When running as a PyInstaller frozen exe, this is the directory containing
    the executable (user-editable config.yaml lives here). When running from
    source, this is the project root (parent of the speechtotext package).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_bundle_root() -> Path:
    """Return the bundle root: exe dir when frozen, or same as get_app_root().
    Use when looking for bundled files that may be in _internal (PyInstaller 6+).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and Path(meipass).is_dir():
            return Path(meipass)
        return exe_dir
    return Path(__file__).resolve().parent.parent
