"""Thread B: Audio producer. Captures microphone with PyAudio, VAD, pushes chunks to queue on stop."""

from __future__ import annotations

import struct
import threading
import time
from queue import Empty, Queue

import pyaudio

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # optional: VAD not used for filtering in this flow

# Whisper expects 16 kHz mono 16-bit PCM
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1


def _frames_to_bytes(frames: list[bytes]) -> bytes:
    return b"".join(frames)


def _bytes_to_float32(data: bytes) -> list[float]:
    """Convert 16-bit PCM bytes to float32 in [-1, 1] for faster-whisper."""
    n = len(data) // 2
    ints = struct.unpack(f"<{n}h", data)
    return [x / 32768.0 for x in ints]


def run_audio_producer(
    *,
    audio_queue: Queue,
    recording_event: threading.Event,
    submit_event: threading.Event,
    stop_event: threading.Event,
    sample_rate: int = SAMPLE_RATE,
    chunk_duration_ms: int = 30,
    vad_aggressiveness: int = 2,
    vad_filter_capture: bool = False,
    partial_audio_queue: Queue | None = None,
    partial_wake_enabled: bool = True,
    partial_wake_first_window_ms: int = 1000,
) -> None:
    """
    Run in a dedicated thread. While recording_event is set, capture audio and buffer it.
    When submit_event is set (on hotkey release), push current buffer to audio_queue and clear buffer.
    When vad_filter_capture is True and webrtcvad is available, only frames classified as speech are kept (reduces background noise).
    Stop when stop_event is set.
    """
    vad = webrtcvad.Vad(vad_aggressiveness) if webrtcvad else None
    use_vad_filter = vad_filter_capture and vad is not None
    use_partial_wake = partial_wake_enabled and partial_audio_queue is not None
    chunk_samples = sample_rate * chunk_duration_ms // 1000
    chunk_bytes = chunk_samples * SAMPLE_WIDTH
    partial_window_bytes = max(chunk_bytes, sample_rate * max(250, int(partial_wake_first_window_ms)) // 1000 * SAMPLE_WIDTH)
    pa = pyaudio.PyAudio()
    stream = None
    buffer: list[bytes] = []
    partial_sent_for_utterance = False
    was_recording = False

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_samples,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open microphone: {e}") from e

    try:
        while not stop_event.is_set():
            if submit_event.is_set():
                if buffer:
                    raw = _frames_to_bytes(buffer)
                    buffer = []
                    audio_queue.put((raw, sample_rate))
                partial_sent_for_utterance = False
                submit_event.clear()

            is_recording = recording_event.is_set()
            if is_recording:
                if not was_recording:
                    partial_sent_for_utterance = False
                try:
                    data = stream.read(chunk_samples, exception_on_overflow=False)
                except Exception:
                    break
                if len(data) == chunk_bytes:
                    if use_vad_filter:
                        if vad.is_speech(data, sample_rate):
                            buffer.append(data)
                    else:
                        buffer.append(data)
                    if use_partial_wake and not partial_sent_for_utterance:
                        total_bytes = sum(len(f) for f in buffer)
                        if total_bytes >= partial_window_bytes:
                            raw_preview = _frames_to_bytes(buffer)
                            partial_audio_queue.put((raw_preview[:partial_window_bytes], sample_rate))
                            partial_sent_for_utterance = True
            else:
                # Not recording: if we have buffered audio, push it and clear
                if buffer:
                    raw = _frames_to_bytes(buffer)
                    buffer = []
                    # faster-whisper accepts numpy or raw; we pass float32 list via numpy later in transcriber
                    # Queue payload: (bytes_raw_16bit, sample_rate) so transcriber can convert
                    audio_queue.put((raw, sample_rate))
                partial_sent_for_utterance = False
                time.sleep(0.02)
            was_recording = is_recording
        if buffer:
            raw = _frames_to_bytes(buffer)
            audio_queue.put((raw, sample_rate))
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()


def start_audio_thread(
    audio_queue: Queue,
    recording_event: threading.Event,
    submit_event: threading.Event,
    stop_event: threading.Event,
    sample_rate: int = SAMPLE_RATE,
    chunk_duration_ms: int = 30,
    vad_aggressiveness: int = 2,
    vad_filter_capture: bool = False,
    partial_audio_queue: Queue | None = None,
    partial_wake_enabled: bool = True,
    partial_wake_first_window_ms: int = 1000,
) -> threading.Thread:
    """Start Thread B. Returns the thread (already started)."""
    thread = threading.Thread(
        target=run_audio_producer,
        kwargs={
            "audio_queue": audio_queue,
            "recording_event": recording_event,
            "submit_event": submit_event,
            "stop_event": stop_event,
            "sample_rate": sample_rate,
            "chunk_duration_ms": chunk_duration_ms,
            "vad_aggressiveness": vad_aggressiveness,
            "vad_filter_capture": vad_filter_capture,
            "partial_audio_queue": partial_audio_queue,
            "partial_wake_enabled": partial_wake_enabled,
            "partial_wake_first_window_ms": partial_wake_first_window_ms,
        },
        name="AudioProducer",
        daemon=True,
    )
    thread.start()
    return thread
