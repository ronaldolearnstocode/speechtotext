#!/usr/bin/env python3
"""
Speech-to-Text MVP: hold global hotkey to record, release to transcribe and type into focused window.
Run: python main.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from queue import Queue

# Add project root so speechtotext package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from speechtotext.audio_capture import start_audio_thread
from speechtotext.config_loader import load_config
from speechtotext.hotkey import start_hotkey_thread
from speechtotext.injector import start_injector_thread
from speechtotext.transcriber import start_transcriber_thread


def main() -> int:
    parser = argparse.ArgumentParser(description="Speech-to-Text: hold hotkey to record, release to transcribe.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Print hotkey/recording events to trace issues")
    args = parser.parse_args()

    config = load_config(args.config)
    hotkey_combo = config["hotkey"]
    quit_hotkey_combo = config.get("quit_hotkey") or "ctrl+shift+q"
    debug = args.debug
    model_name = config["model_name"]
    device = config["device"]
    compute_type = config["compute_type"]
    sample_rate = config["sample_rate"]
    chunk_duration_ms = config["chunk_duration_ms"]
    vad_aggressiveness = config["vad_aggressiveness"]
    type_interval = config["type_interval"]

    audio_queue: Queue = Queue()
    text_queue: Queue = Queue()
    recording_event = threading.Event()
    stop_event = threading.Event()

    def on_press() -> None:
        recording_event.set()
        if debug:
            print("[main] recording_event SET")

    def on_release() -> None:
        recording_event.clear()
        if debug:
            print("[main] recording_event CLEAR (audio should be submitted)")

    # Start threads: B (audio), C (transcriber), D (injector), then A (hotkey)
    t_audio = start_audio_thread(
        audio_queue=audio_queue,
        recording_event=recording_event,
        stop_event=stop_event,
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        vad_aggressiveness=vad_aggressiveness,
    )
    t_transcriber = start_transcriber_thread(
        audio_queue=audio_queue,
        text_queue=text_queue,
        stop_event=stop_event,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    t_injector = start_injector_thread(
        text_queue=text_queue,
        stop_event=stop_event,
        interval=type_interval,
    )
    t_hotkey = start_hotkey_thread(
        hotkey_combo=hotkey_combo,
        on_press=on_press,
        on_release=on_release,
        stop_event=stop_event,
        debug=debug,
        quit_hotkey_combo=quit_hotkey_combo,
    )

    def shutdown(*args: object) -> None:
        stop_event.set()
        recording_event.clear()
        audio_queue.put(None)
        text_queue.put(None)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    print(f"Speech-to-Text running. Hold {hotkey_combo} to record, release to transcribe. Press {quit_hotkey_combo} or Ctrl+C to exit.")
    if debug:
        print("[debug] If hotkey does nothing, try: Run terminal as Administrator (Windows needs this for global hotkeys).")
    t_hotkey.join()
    t_audio.join(timeout=2.0)
    t_transcriber.join(timeout=5.0)
    t_injector.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
