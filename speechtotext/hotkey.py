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
    debug: bool = False,
    quit_hotkey_combo: str | None = None,
) -> None:
    """
    Run the hotkey listener in the current thread. Blocks until stop_event is set.
    on_press: called when hotkey combo is pressed (start recording).
    on_release: called when any key in combo is released (stop and submit).
    quit_hotkey_combo: when pressed, sets stop_event to exit the app (e.g. "ctrl+shift+q").
    """
    combo = parse_hotkey(hotkey_combo)
    want = _combo_keys(combo)
    quit_want: set[str] | None = None
    quit_down: set[str] = set()
    if quit_hotkey_combo:
        quit_want = _combo_keys(parse_hotkey(quit_hotkey_combo))
    state: dict = {"down": set(), "recording": False}

    # Windows: keyboard lib may report keys with "left"/"right" prefix or different names
    def normalize_name(name: str) -> str:
        n = (name or "").lower().strip()
        if n in ("windows", "win", "cmd", "super", "left windows", "right windows"):
            return "windows"
        if n in ("control", "ctrl", "left ctrl", "right ctrl", "left control", "right control"):
            return "ctrl"
        if n in ("shift", "left shift", "right shift"):
            return "shift"
        if n in ("alt", "alt gr", "left alt", "right alt"):
            return "alt"
        return n

    def on_key_event(e: keyboard.KeyboardEvent) -> None:
        key = normalize_name(e.name)
        # Quit hotkey: pressing the combo sets stop_event
        if quit_want and key in quit_want:
            if e.event_type == "down":
                quit_down.add(key)
                if quit_down >= quit_want:
                    if debug:
                        print("[hotkey] QUIT pressed, exiting...")
                    stop_event.set()
            else:
                quit_down.discard(key)
            return
        if key not in want:
            return
        # Debug: print only on new key press (not on key repeat while holding)
        if debug and e.event_type == "down" and key not in state["down"]:
            print(f"[hotkey] down: raw_name={e.name!r} -> key={key!r} down={state['down']!r} want={want!r}")
        if e.event_type == "down":
            state["down"].add(key)
            # On Windows the hook sometimes misses Ctrl when Ctrl+Alt is pressed; check is_pressed
            if state["down"] >= want and not state["recording"]:
                state["recording"] = True
                if debug:
                    print("[hotkey] RECORDING START")
                on_press()
            elif not state["recording"] and "ctrl" in want and "alt" in want:
                if key == "alt" and keyboard.is_pressed("ctrl"):
                    state["down"].add("ctrl")
                    if state["down"] >= want:
                        state["recording"] = True
                        if debug:
                            print("[hotkey] RECORDING START (ctrl detected via is_pressed)")
                        on_press()
        else:
            if state["recording"]:
                state["recording"] = False
                if debug:
                    print("[hotkey] RECORDING STOP (submitting)")
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
    debug: bool = False,
    quit_hotkey_combo: str | None = None,
) -> threading.Thread:
    """Start Thread A in a background thread. Returns the thread (already started)."""
    thread = threading.Thread(
        target=run_hotkey_listener,
        args=(hotkey_combo, on_press, on_release, stop_event),
        kwargs={"debug": debug, "quit_hotkey_combo": quit_hotkey_combo},
        name="HotkeyListener",
        daemon=True,
    )
    thread.start()
    return thread
