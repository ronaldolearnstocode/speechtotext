#!/usr/bin/env python3
"""
Speech-to-Text MVP: hold global hotkey to record, release to transcribe and type into focused window.
Run: python main.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
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
from speechtotext.assistant_router import start_router_thread
from speechtotext.assistant_worker import start_assistant_thread
from speechtotext.assistant_output_window import start_output_window_thread
from speechtotext.tts import play_ack_beep, stop_speaking
from speechtotext.transcriber import get_cuda_bin_path, start_transcriber_thread


def _acquire_single_instance_lock(port: int = 59337) -> socket.socket | None:
    """Ensure only one app instance types at a time."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        try:
            sock.close()
        except Exception:
            pass
        return None
    sock.listen(1)
    return sock


def _has_existing_stt_process() -> bool:
    """Detect other running STT processes on Windows, including older builds."""
    if sys.platform != "win32":
        return False
    pid = os.getpid()
    script = (
        "$p = Get-CimInstance Win32_Process | Where-Object { "
        "$_.ProcessId -ne " + str(pid) + " -and ("
        "$_.Name -ieq 'SpeechToText.exe' -or "
        "($_.Name -ieq 'python.exe' -and $_.CommandLine -match 'main.py') -or "
        "($_.Name -ieq 'pythonw.exe' -and $_.CommandLine -match 'main.py')"
        ") }; "
        "if ($p) { '1' } else { '0' }"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=flags,
        )
        return proc.returncode == 0 and proc.stdout.strip().endswith("1")
    except Exception:
        return False


def main() -> int:
    if _has_existing_stt_process():
        print("[main] Another Speech-to-Text process is already running. Close it first to avoid duplicate typing.")
        return 1

    instance_lock = _acquire_single_instance_lock()
    if instance_lock is None:
        print("[main] Another Speech-to-Text instance is already running. Close it first to avoid duplicate typing.")
        return 1

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
    assistant_enabled = bool(config.get("assistant_enabled", True))
    assistant_wake_word_map = config.get("assistant_wake_word_map", {"command": "ollama", "mother": "gemini"})
    if not isinstance(assistant_wake_word_map, dict):
        assistant_wake_word_map = {"command": "ollama", "mother": "gemini"}
    assistant_provider_enabled = config.get("assistant_provider_enabled", {"ollama": True, "gemini": False})
    if not isinstance(assistant_provider_enabled, dict):
        assistant_provider_enabled = {"ollama": True, "gemini": False}
    assistant_mode = str(config.get("assistant_mode", "work")).strip().lower()
    assistant_model_work = str(config.get("assistant_model_work", "qwen2.5-coder:7b")).strip()
    assistant_model_quality = str(config.get("assistant_model_quality", "qwen2.5-coder:14b")).strip()
    assistant_ollama_host = str(config.get("assistant_ollama_host", "http://127.0.0.1:11434")).strip()
    assistant_timeout_s = float(config.get("assistant_timeout_s", 60))
    assistant_temperature = float(config.get("assistant_temperature", 0.3))
    assistant_max_tokens = int(config.get("assistant_max_tokens", 256))
    assistant_tts_provider = str(config.get("assistant_tts_provider", "windows")).strip().lower()
    assistant_tts_voice = str(config.get("assistant_tts_voice", "male")).strip()
    assistant_tts_rate = int(config.get("assistant_tts_rate", 0))
    assistant_tts_volume = int(config.get("assistant_tts_volume", 100))
    assistant_tts_piper_path = str(config.get("assistant_tts_piper_path", "")).strip()
    assistant_tts_piper_model = str(config.get("assistant_tts_piper_model", "")).strip()
    assistant_output_window_enabled = bool(config.get("assistant_output_window_enabled", True))
    assistant_output_window_topmost = bool(config.get("assistant_output_window_topmost", False))
    assistant_voice_summary_only = bool(config.get("assistant_voice_summary_only", True))
    assistant_voice_summary_max_chars = int(config.get("assistant_voice_summary_max_chars", 160))
    partial_wake_enabled = bool(config.get("assistant_partial_wake_enabled", True))
    partial_wake_first_window_ms = int(config.get("assistant_partial_wake_first_window_ms", 1000))

    audio_queue: Queue = Queue()
    partial_audio_queue: Queue = Queue()
    raw_text_queue: Queue = Queue()
    inject_text_queue: Queue = Queue()
    assistant_query_queue: Queue = Queue()
    assistant_output_queue: Queue = Queue()
    recording_event = threading.Event()
    submit_recording_event = threading.Event()
    early_wake_beep_event = threading.Event()
    stop_event = threading.Event()
    device_state = {"device": device, "compute_type": compute_type}
    reload_event = threading.Event()
    cuda_available = bool(get_cuda_bin_path()) if sys.platform == "win32" else False

    def on_press() -> None:
        # If assistant is speaking and user starts recording, stop speech immediately
        # so playback does not leak into the new mic capture.
        if stop_speaking() and debug:
            print("[assistant] speech stopped by hotkey press")
        early_wake_beep_event.clear()
        recording_event.set()
        if debug:
            print("[main] recording_event SET")

    def on_release() -> None:
        # Interrupt assistant speech when hotkey is released.
        if stop_speaking():
            if debug:
                print("[assistant] speech stopped by hotkey release")
        submit_recording_event.set()
        recording_event.clear()
        # Distinct low-pitch short beep = recording stopped.
        play_ack_beep(frequency_hz=700, duration_ms=60)
        if debug:
            print("[main] recording_event CLEAR (audio should be submitted)")

    tray_icon = None
    tray_thread: threading.Thread | None = None
    output_window_thread: threading.Thread | None = None
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
        submit_event=submit_recording_event,
        stop_event=stop_event,
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        vad_aggressiveness=vad_aggressiveness,
        vad_filter_capture=vad_filter_capture,
        partial_audio_queue=partial_audio_queue,
        partial_wake_enabled=partial_wake_enabled and assistant_enabled,
        partial_wake_first_window_ms=partial_wake_first_window_ms,
    )
    t_transcriber = start_transcriber_thread(
        audio_queue=audio_queue,
        text_queue=raw_text_queue,
        stop_event=stop_event,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        vad_filter=vad_filter,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        report_timing=args.cpu or args.gpu,
        device_label=device,
        partial_audio_queue=partial_audio_queue,
        partial_wake_enabled=partial_wake_enabled and assistant_enabled,
        partial_wake_word_map=assistant_wake_word_map,
        early_wake_beep_event=early_wake_beep_event,
        device_state=device_state,
        reload_event=reload_event,
        on_ready_changed=lambda ready: on_ready_changed_ref[0](ready),
        on_device_changed=lambda: on_device_changed_ref[0](),
    )
    t_router = start_router_thread(
        raw_text_queue=raw_text_queue,
        inject_text_queue=inject_text_queue,
        assistant_query_queue=assistant_query_queue,
        stop_event=stop_event,
        assistant_enabled=assistant_enabled,
        wake_word_map=assistant_wake_word_map,
        early_wake_beep_event=early_wake_beep_event,
        device_state=device_state,
        debug=debug,
    )
    t_assistant = start_assistant_thread(
        assistant_query_queue=assistant_query_queue,
        stop_event=stop_event,
        provider_enabled=assistant_provider_enabled,
        ollama_host=assistant_ollama_host,
        model_work=assistant_model_work,
        model_quality=assistant_model_quality,
        mode=assistant_mode,
        timeout_s=assistant_timeout_s,
        temperature=assistant_temperature,
        max_tokens=assistant_max_tokens,
        tts_provider=assistant_tts_provider,
        tts_voice=assistant_tts_voice,
        tts_rate=assistant_tts_rate,
        tts_volume=assistant_tts_volume,
        tts_piper_path=assistant_tts_piper_path,
        tts_piper_model=assistant_tts_piper_model,
        assistant_output_queue=assistant_output_queue if assistant_output_window_enabled else None,
        voice_summary_only=assistant_voice_summary_only,
        voice_summary_max_chars=assistant_voice_summary_max_chars,
        debug=debug,
    )
    if assistant_enabled and assistant_output_window_enabled:
        output_window_thread = start_output_window_thread(
            output_queue=assistant_output_queue,
            stop_event=stop_event,
            always_on_top=assistant_output_window_topmost,
        )
    t_injector = start_injector_thread(
        text_queue=inject_text_queue,
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
        submit_recording_event.set()
        audio_queue.put(None)
        partial_audio_queue.put(None)
        raw_text_queue.put(None)
        inject_text_queue.put(None)
        assistant_query_queue.put(None)
        assistant_output_queue.put(None)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    print(f"Speech-to-Text running. Hold {hotkey_combo} to record, release to transcribe. Press {quit_hotkey_combo} or Ctrl+C to exit.")
    if args.cpu or args.gpu:
        print(f"Device: {device} ({compute_type}). Per-transcription time will be printed.")
    if not args.no_tray:
        print("Tray icon active: green = ready, red = loading. Right-click for Show/Hide, Use CPU/GPU, or Quit.")
    if assistant_enabled:
        print("[assistant] wake words:", ", ".join(sorted(str(k) for k in assistant_wake_word_map.keys())))
        print(f"[assistant] mode={assistant_mode}, local model={assistant_model_work if assistant_mode != 'quality' else assistant_model_quality}")
    if debug:
        print("[debug] If hotkey does nothing, try: Run terminal as Administrator (Windows needs this for global hotkeys).")
    t_hotkey.join()
    if tray_icon is not None:
        tray_icon.stop()
    if tray_thread is not None:
        tray_thread.join(timeout=2.0)
    t_audio.join(timeout=2.0)
    t_transcriber.join(timeout=5.0)
    t_router.join(timeout=2.0)
    t_assistant.join(timeout=2.0)
    if output_window_thread is not None:
        output_window_thread.join(timeout=2.0)
    t_injector.join(timeout=2.0)
    try:
        instance_lock.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
