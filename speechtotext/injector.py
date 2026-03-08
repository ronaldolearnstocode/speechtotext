"""Thread D: Injector. Consumes text from queue and types it into the focused window via pyautogui."""

from __future__ import annotations

import threading
from queue import Empty, Queue

import pyautogui

# Disable failsafe (moving mouse to corner to abort) for an always-on tool
pyautogui.FAILSAFE = False


def run_injector(
    text_queue: Queue,
    stop_event: threading.Event,
    interval: float = 0.02,
    timeout: float = 1.0,
) -> None:
    """
    Loop: get text from text_queue, then pyautogui.write(text, interval=interval).
    Stops when stop_event is set.
    """
    while not stop_event.is_set():
        try:
            text = text_queue.get(timeout=timeout)
        except Empty:
            continue
        if text is None:
            break
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            pyautogui.write(text, interval=interval)
        except Exception:
            pass


def start_injector_thread(
    text_queue: Queue,
    stop_event: threading.Event,
    interval: float = 0.02,
) -> threading.Thread:
    """Start Thread D. Returns the thread (already started)."""
    thread = threading.Thread(
        target=run_injector,
        kwargs={
            "text_queue": text_queue,
            "stop_event": stop_event,
            "interval": interval,
            "timeout": 1.0,
        },
        name="Injector",
        daemon=True,
    )
    thread.start()
    return thread
