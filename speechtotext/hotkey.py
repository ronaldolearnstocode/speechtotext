"""Thread A: Global hotkey listener. Signals start/stop recording to the audio producer."""

from __future__ import annotations

import threading
from typing import Callable

import keyboard


def parse_hotkey(hotkey: str) -> str:
    """Normalize hotkey string for the keyboard library (e.g. 'ctrl+win' -> 'ctrl+windows')."""
    s = hotkey.strip().lower().replace(" ", "")
    if "win" in s and "windows" not in s:
        s = s.replace("win", "windows")
    return s


def _combo_keys(combo: str) -> set[str]:
    """Return set of key names in the combo (e.g. 'ctrl+windows' -> {'ctrl', 'windows'})."""
    return set(k.strip().lower() for k in combo.split("+") if k.strip())


def run_hotkey_listener(
    hotkey_combo: str,
    on_press: Callable[[], None],
    on_release: Callable[[], None],
    stop_event: threading.Event,
) -> None:
    """
    Run the hotkey listener in the current thread. Blocks until stop_event is set.
    on_press: called when hotkey combo is pressed (start recording).
    on_release: called when any key in combo is released (stop and submit).
    """
    combo = parse_hotkey(hotkey_combo)
    want = _combo_keys(combo)
    state: dict = {"down": set(), "recording": False}

    def normalize_name(name: str) -> str:
        n = (name or "").lower()
        if n == "windows" or n == "win":
            return "windows"
        return n

    def on_key_event(e: keyboard.KeyboardEvent) -> None:
        key = normalize_name(e.name)
        if key not in want:
            return
        if e.event_type == "down":
            state["down"].add(key)
            if state["down"] >= want and not state["recording"]:
                state["recording"] = True
                on_press()
        else:
            if state["recording"]:
                state["recording"] = False
                on_release()
            state["down"].discard(key)

    keyboard.hook(on_key_event)
    try:
        stop_event.wait()
    finally:
        keyboard.unhook_all()


def start_hotkey_thread(
    hotkey_combo: str,
    on_press: Callable[[], None],
    on_release: Callable[[], None],
    stop_event: threading.Event,
) -> threading.Thread:
    """Start Thread A in a background thread. Returns the thread (already started)."""
    thread = threading.Thread(
        target=run_hotkey_listener,
        args=(hotkey_combo, on_press, on_release, stop_event),
        name="HotkeyListener",
        daemon=True,
    )
    thread.start()
    return thread
