"""Thread C: AI consumer. Loads faster-whisper once, consumes audio from queue, pushes text to injector queue."""

from __future__ import annotations

import os
import struct
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

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
) -> None:
    """
    Load Whisper model once, then loop: get (raw_bytes, sample_rate) from audio_queue,
    transcribe, put text into text_queue. Use timeout on get to check stop_event.
    When vad_filter is True, only speech segments are transcribed (reduces background noise).
    """
    # On Windows with CUDA, add CUDA bin to DLL search path so cublas64_12.dll etc. can be loaded
    if device == "cuda" and sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        cuda_bin = get_cuda_bin_path()
        if cuda_bin:
            os.add_dll_directory(cuda_bin)
    WhisperModel = _get_whisper()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    while not stop_event.is_set():
        try:
            item = audio_queue.get(timeout=timeout)
        except Empty:
            continue
        if item is None:
            break
        raw_bytes, sample_rate = item
        if not raw_bytes:
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
            print(f"Transcription ({device_label}): {elapsed:.2f} s")
        if text:
            text_queue.put(text)


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
) -> threading.Thread:
    """Start Thread C. Model is loaded inside the thread to avoid blocking main."""
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
        },
        name="Transcriber",
        daemon=True,
    )
    thread.start()
    return thread
