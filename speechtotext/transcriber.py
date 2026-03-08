"""Thread C: AI consumer. Loads faster-whisper once, consumes audio from queue, pushes text to injector queue."""

from __future__ import annotations

import struct
import threading
from queue import Empty, Queue

import numpy as np

# Lazy import to avoid loading heavy model until needed
_WhisperModel = None

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
    timeout: float = 1.0,
) -> None:
    """
    Load Whisper model once, then loop: get (raw_bytes, sample_rate) from audio_queue,
    transcribe, put text into text_queue. Use timeout on get to check stop_event.
    """
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
        segments, info = model.transcribe(audio_f32, language=language, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
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
        },
        name="Transcriber",
        daemon=True,
    )
    thread.start()
    return thread
