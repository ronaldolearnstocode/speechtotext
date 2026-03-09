"""Thread C: AI consumer. Loads faster-whisper once, consumes audio from queue, pushes text to injector queue."""

from __future__ import annotations

import os
import struct
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

import numpy as np
from speechtotext.assistant_router import parse_assistant_wake
from speechtotext.tts import play_ack_beep

# Lazy import to avoid loading heavy model until needed
_WhisperModel = None


# Re-export for callers that import from transcriber
from speechtotext.cuda_path import get_cuda_bin_path


def _get_whisper():
    global _WhisperModel
    if _WhisperModel is None:
        from faster_whisper import WhisperModel as WM
        _WhisperModel = WM
    return _WhisperModel


def _bytes_to_f32(data: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 in [-1, 1]."""
    n = len(data) // 2
    ints = struct.unpack(f"<{n}h", data)
    return np.array(ints, dtype=np.float32) / 32768.0


def _load_model(
    WhisperModel,
    model_name: str,
    device: str,
    compute_type: str,
) -> "object":
    """Load Whisper model; add CUDA bin to DLL path on Windows when device is cuda."""
    if device == "cuda" and sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        cuda_bin = get_cuda_bin_path()
        if cuda_bin:
            os.add_dll_directory(cuda_bin)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def run_transcriber(
    *,
    audio_queue: Queue,
    text_queue: Queue,
    stop_event: threading.Event,
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    vad_filter: bool = True,
    vad_min_silence_duration_ms: int = 500,
    report_timing: bool = False,
    device_label: str = "cpu",
    timeout: float = 1.0,
    partial_audio_queue: Queue | None = None,
    partial_wake_enabled: bool = True,
    partial_wake_word_map: dict | None = None,
    early_wake_beep_event: threading.Event | None = None,
    device_state: dict | None = None,
    reload_event: threading.Event | None = None,
    on_ready_changed: Callable[[bool], None] | None = None,
    on_device_changed: Callable[[], None] | None = None,
) -> None:
    """
    Load Whisper model once, then loop: get (raw_bytes, sample_rate) from audio_queue,
    transcribe, put text into text_queue. If device_state and reload_event are provided,
    supports runtime device switch: when reload_event is set, reloads model from device_state.
    on_ready_changed(False/True) signals loading vs ready; on_device_changed() refreshes tray menu.
    """
    use_reload = device_state is not None and reload_event is not None
    if on_ready_changed is None:
        on_ready_changed = lambda _: None
    if on_device_changed is None:
        on_device_changed = lambda: None

    WhisperModel = _get_whisper()
    current_device = device
    current_compute_type = compute_type
    current_label = device_label

    def load_current_model():
        nonlocal current_device, current_compute_type, current_label
        if use_reload and device_state:
            current_device = device_state.get("device", "cpu")
            current_compute_type = device_state.get("compute_type", "int8")
            current_label = current_device
        return _load_model(WhisperModel, model_name, current_device, current_compute_type)

    on_ready_changed(False)
    try:
        model = load_current_model()
    except Exception:
        on_ready_changed(True)
        raise
    on_ready_changed(True)

    while not stop_event.is_set():
        assistant_on_gpu = not (device_state is not None and device_state.get("device") == "cpu")
        use_partial_wake = partial_wake_enabled and partial_audio_queue is not None and assistant_on_gpu
        if use_partial_wake:
            try:
                partial_item = partial_audio_queue.get_nowait()
            except Empty:
                partial_item = None
            if partial_item is not None and isinstance(partial_item, tuple) and len(partial_item) == 2:
                partial_bytes, _partial_sr = partial_item
                can_beep = early_wake_beep_event is None or not early_wake_beep_event.is_set()
                if can_beep and partial_bytes and model is not None:
                    partial_audio_f32 = _bytes_to_f32(partial_bytes)
                    try:
                        partial_segments, _ = model.transcribe(
                            partial_audio_f32,
                            language=language,
                            beam_size=1,
                            vad_filter=False,
                        )
                        partial_text = " ".join(s.text for s in partial_segments).strip()
                    except Exception:
                        partial_text = ""
                    parsed = parse_assistant_wake(
                        partial_text,
                        partial_wake_word_map,
                        assistant_enabled=True,
                    )
                    if parsed is not None:
                        if play_ack_beep() and early_wake_beep_event is not None:
                            early_wake_beep_event.set()
                        elif early_wake_beep_event is not None:
                            early_wake_beep_event.set()
                        if report_timing:
                            print("[assistant] early wake detected from first-second partial")

        if use_reload and reload_event.is_set():
            on_ready_changed(False)
            model = None
            try:
                if device_state:
                    current_device = device_state.get("device", "cpu")
                    current_compute_type = device_state.get("compute_type", "int8")
                    current_label = current_device
                model = _load_model(WhisperModel, model_name, current_device, current_compute_type)
            except RuntimeError as e:
                err = str(e).lower()
                if "cublas" in err or "cuda" in err or ".dll" in err or "not found" in err:
                    print("[transcriber] GPU/CUDA error:", e)
                    print("[transcriber] Use CPU from the tray menu or install CUDA 12.")
                # Fall back to CPU so app stays usable
                if device_state:
                    device_state["device"] = "cpu"
                    device_state["compute_type"] = "int8"
                    current_device = "cpu"
                    current_compute_type = "int8"
                    current_label = "cpu"
                try:
                    model = _load_model(WhisperModel, model_name, "cpu", "int8")
                except Exception:
                    pass
            finally:
                reload_event.clear()
                on_device_changed()
                on_ready_changed(True)

        wait_timeout = min(timeout, 0.1) if use_partial_wake else timeout
        try:
            item = audio_queue.get(timeout=wait_timeout)
        except Empty:
            continue
        if item is None:
            break
        raw_bytes, sample_rate = item
        if not raw_bytes:
            continue
        if model is None:
            continue
        audio_f32 = _bytes_to_f32(raw_bytes)
        transcribe_kw: dict = {
            "language": language,
            "beam_size": 1,
        }
        if vad_filter:
            transcribe_kw["vad_filter"] = True
            transcribe_kw["vad_parameters"] = dict(min_silence_duration_ms=vad_min_silence_duration_ms)
        t0 = time.perf_counter() if report_timing else None
        try:
            segments, info = model.transcribe(audio_f32, **transcribe_kw)
            text = " ".join(s.text for s in segments).strip()
        except RuntimeError as e:
            err = str(e).lower()
            if "cublas" in err or "cuda" in err or ".dll" in err or "not found" in err:
                print("[transcriber] GPU/CUDA error:", e)
                print("[transcriber] This build expects CUDA 12 (cublas64_12.dll). If you have only CUDA 13, install CUDA 12 alongside or use --cpu.")
                break
            raise
        if report_timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            print(f"Transcription ({current_label}): {elapsed:.2f} s")
        if text:
            text_queue.put(text)

    on_ready_changed(True)


def start_transcriber_thread(
    audio_queue: Queue,
    text_queue: Queue,
    stop_event: threading.Event,
    model_name: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    vad_filter: bool = True,
    vad_min_silence_duration_ms: int = 500,
    report_timing: bool = False,
    device_label: str = "cpu",
    partial_audio_queue: Queue | None = None,
    partial_wake_enabled: bool = True,
    partial_wake_word_map: dict | None = None,
    early_wake_beep_event: threading.Event | None = None,
    device_state: dict | None = None,
    reload_event: threading.Event | None = None,
    on_ready_changed: Callable[[bool], None] | None = None,
    on_device_changed: Callable[[], None] | None = None,
) -> threading.Thread:
    """Start Thread C. Model is loaded inside the thread. Supports runtime device switch if device_state/reload_event provided."""
    thread = threading.Thread(
        target=run_transcriber,
        kwargs={
            "audio_queue": audio_queue,
            "text_queue": text_queue,
            "stop_event": stop_event,
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "language": language,
            "vad_filter": vad_filter,
            "vad_min_silence_duration_ms": vad_min_silence_duration_ms,
            "report_timing": report_timing,
            "device_label": device_label,
            "partial_audio_queue": partial_audio_queue,
            "partial_wake_enabled": partial_wake_enabled,
            "partial_wake_word_map": partial_wake_word_map,
            "early_wake_beep_event": early_wake_beep_event,
            "device_state": device_state,
            "reload_event": reload_event,
            "on_ready_changed": on_ready_changed,
            "on_device_changed": on_device_changed,
        },
        name="Transcriber",
        daemon=True,
    )
    thread.start()
    return thread
