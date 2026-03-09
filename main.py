#!/usr/bin/env python3
"""
Speech-to-Text MVP: hold global hotkey to record, release to transcribe and type into focused window.
Run: python main.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from queue import Queue

# Add project root so speechtotext package is importable (not needed when frozen)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Parse args and add CUDA bin to DLL path *before* importing transcriber (which pulls in numpy/ctranslate2)
parser = argparse.ArgumentParser(description="Speech-to-Text: hold hotkey to record, release to transcribe.")
parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
parser.add_argument("--debug", action="store_true", help="Print hotkey/recording events to trace issues")
parser.add_argument("--no-tray", action="store_true", help="Run without system tray icon")
device_group = parser.add_mutually_exclusive_group()
device_group.add_argument("--cpu", action="store_true", help="Force CPU (int8) for transcription")
device_group.add_argument("--gpu", action="store_true", help="Force GPU/CUDA (float16) for transcription; overrides config device/compute_type")
args = parser.parse_args()

if args.gpu and sys.platform == "win32":
    from speechtotext import cuda_path
    cuda_bin = cuda_path.get_cuda_bin_path()
    if cuda_bin:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cuda_bin)
        # Prepend to PATH so loaders that use PATH (e.g. ctranslate2) find cublas64_12.dll
        path_env = os.environ.get("PATH", "")
        if cuda_bin not in path_env.split(os.pathsep):
            os.environ["PATH"] = cuda_bin + os.pathsep + path_env

from speechtotext.audio_capture import start_audio_thread
from speechtotext.config_loader import load_config
from speechtotext.hotkey import start_hotkey_thread
from speechtotext.injector import start_injector_thread
from speechtotext.transcriber import get_cuda_bin_path, start_transcriber_thread


def main() -> int:
    config = load_config(args.config)
    hotkey_combo = config["hotkey"]
    quit_hotkey_combo = config.get("quit_hotkey") or "ctrl+shift+q"
    debug = args.debug
    model_name = config["model_name"]
    device = config["device"]
    compute_type = config["compute_type"]
    if args.cpu:
        device = "cpu"
        compute_type = "int8"
    elif args.gpu:
        device = "cuda"
        compute_type = "float16"
        cuda_bin = get_cuda_bin_path()
        if not cuda_bin or not Path(cuda_bin).is_dir():
            print("[main] CUDA not found; --gpu will likely fail. Set CUDA_PATH or add the toolkit bin folder to PATH.")
        else:
            if debug:
                print("[main] CUDA bin added to DLL path:", cuda_bin)
    sample_rate = config["sample_rate"]
    chunk_duration_ms = config["chunk_duration_ms"]
    vad_aggressiveness = config["vad_aggressiveness"]
    vad_filter = config.get("vad_filter", True)
    vad_min_silence_duration_ms = config.get("vad_min_silence_duration_ms", 500)
    vad_filter_capture = config.get("vad_filter_capture", False)
    type_interval = config["type_interval"]
    show_window_on_start = config.get("show_window_on_start", False)

    audio_queue: Queue = Queue()
    text_queue: Queue = Queue()
    recording_event = threading.Event()
    stop_event = threading.Event()
    device_state = {"device": device, "compute_type": compute_type}
    reload_event = threading.Event()
    cuda_available = bool(get_cuda_bin_path()) if sys.platform == "win32" else False

    def on_press() -> None:
        recording_event.set()
        if debug:
            print("[main] recording_event SET")

    def on_release() -> None:
        recording_event.clear()
        if debug:
            print("[main] recording_event CLEAR (audio should be submitted)")

    tray_icon = None
    tray_thread: threading.Thread | None = None
    on_ready_changed_ref: list = [lambda _: None]
    on_device_changed_ref: list = [lambda: None]

    if not args.no_tray:
        from speechtotext.tray import create_tray_icon, create_icon_image
        menu_update_callback_ref = [None]
        tray_icon = create_tray_icon(
            stop_event,
            show_window_on_start=show_window_on_start,
            device_state=device_state,
            reload_event=reload_event,
            menu_update_callback_ref=menu_update_callback_ref,
            cuda_available=cuda_available,
        )
        menu_update_callback_ref[0] = lambda: tray_icon.update_menu()
        def _on_ready(ready: bool) -> None:
            try:
                tray_icon.icon = create_icon_image(64, ready=ready)
            except Exception:
                pass
        on_ready_changed_ref[0] = _on_ready
        on_device_changed_ref[0] = lambda: tray_icon.update_menu()
        tray_thread = threading.Thread(target=tray_icon.run, daemon=True)
        tray_thread.start()

    # Start threads: B (audio), C (transcriber), D (injector), then A (hotkey)
    t_audio = start_audio_thread(
        audio_queue=audio_queue,
        recording_event=recording_event,
        stop_event=stop_event,
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        vad_aggressiveness=vad_aggressiveness,
        vad_filter_capture=vad_filter_capture,
    )
    t_transcriber = start_transcriber_thread(
        audio_queue=audio_queue,
        text_queue=text_queue,
        stop_event=stop_event,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        vad_filter=vad_filter,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        report_timing=args.cpu or args.gpu,
        device_label=device,
        device_state=device_state,
        reload_event=reload_event,
        on_ready_changed=lambda ready: on_ready_changed_ref[0](ready),
        on_device_changed=lambda: on_device_changed_ref[0](),
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
    if args.cpu or args.gpu:
        print(f"Device: {device} ({compute_type}). Per-transcription time will be printed.")
    if not args.no_tray:
        print("Tray icon active: green = ready, red = loading. Right-click for Show/Hide, Use CPU/GPU, or Quit.")
    if debug:
        print("[debug] If hotkey does nothing, try: Run terminal as Administrator (Windows needs this for global hotkeys).")
    t_hotkey.join()
    if tray_icon is not None:
        tray_icon.stop()
    if tray_thread is not None:
        tray_thread.join(timeout=2.0)
    t_audio.join(timeout=2.0)
    t_transcriber.join(timeout=5.0)
    t_injector.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
