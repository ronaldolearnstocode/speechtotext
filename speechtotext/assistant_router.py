"""Wake-word router: dispatch transcribed text to injector or assistant queue."""

from __future__ import annotations

import threading
from queue import Empty, Queue

from speechtotext.assistant_worker import _append_speech_to_text_log
from speechtotext.tts import play_ack_beep


def _normalize_wake_word_map(wake_word_map: dict | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for k, v in (wake_word_map or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        key = k.strip().lower()
        value = v.strip().lower()
        if key and value:
            result[key] = value
    return result


def _parse_wake_word(text: str, wake_word_map: dict[str, str]) -> tuple[str, str, str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    lower = stripped.lower()
    for wake_word, provider in wake_word_map.items():
        if not lower.startswith(wake_word):
            continue
        # Trigger only when wake word is at the start and not followed by another word char.
        if len(lower) > len(wake_word) and lower[len(wake_word)].isalnum():
            continue
        remainder = stripped[len(wake_word):].lstrip(" \t,;:.!?-")
        return wake_word, provider, remainder.strip()
    return None


def parse_assistant_wake(
    text: str,
    wake_word_map: dict | None,
    *,
    assistant_enabled: bool = True,
) -> tuple[str, str, str] | None:
    """Parse assistant wake word from text using router-compatible rules."""
    if not assistant_enabled:
        return None
    normalized_map = _normalize_wake_word_map(wake_word_map)
    if not normalized_map:
        return None
    return _parse_wake_word(text, normalized_map)


def run_router(
    *,
    raw_text_queue: Queue,
    inject_text_queue: Queue,
    assistant_query_queue: Queue,
    stop_event: threading.Event,
    assistant_enabled: bool,
    wake_word_map: dict | None,
    early_wake_beep_event: threading.Event | None = None,
    device_state: dict | None = None,
    log_dir=None,
    timeout: float = 1.0,
    debug: bool = False,
) -> None:
    """Route transcribed text: wake-word commands to assistant, everything else to injector."""
    while not stop_event.is_set():
        try:
            text = raw_text_queue.get(timeout=timeout)
        except Empty:
            continue
        if text is None:
            inject_text_queue.put(None)
            assistant_query_queue.put(None)
            break
        if not isinstance(text, str) or not text.strip():
            continue
        if log_dir is not None:
            _append_speech_to_text_log(log_dir, text)

        assistant_active = assistant_enabled and not (
            device_state is not None and device_state.get("device") == "cpu"
        )
        parsed = parse_assistant_wake(text, wake_word_map, assistant_enabled=assistant_active)
        if parsed is None:
            inject_text_queue.put(text)
            continue

        wake_word, provider, prompt = parsed
        if not prompt:
            if debug:
                print(f"[assistant] wake word detected without prompt: {wake_word}")
            continue
        # Quick audible confirmation that assistant mode was triggered.
        beep_already_played = early_wake_beep_event is not None and early_wake_beep_event.is_set()
        if not beep_already_played:
            play_ack_beep()
            if early_wake_beep_event is not None:
                early_wake_beep_event.set()
        payload = {
            "provider": provider,
            "wake_word": wake_word,
            "prompt": prompt,
            "raw_text": text,
        }
        if debug:
            print(f"[assistant] wake word detected: {wake_word} -> {provider}")
        assistant_query_queue.put(payload)


def start_router_thread(
    raw_text_queue: Queue,
    inject_text_queue: Queue,
    assistant_query_queue: Queue,
    stop_event: threading.Event,
    assistant_enabled: bool,
    wake_word_map: dict | None,
    early_wake_beep_event: threading.Event | None = None,
    device_state: dict | None = None,
    log_dir=None,
    debug: bool = False,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_router,
        kwargs={
            "raw_text_queue": raw_text_queue,
            "inject_text_queue": inject_text_queue,
            "assistant_query_queue": assistant_query_queue,
            "stop_event": stop_event,
            "assistant_enabled": assistant_enabled,
            "wake_word_map": wake_word_map,
            "early_wake_beep_event": early_wake_beep_event,
            "device_state": device_state,
            "log_dir": log_dir,
            "timeout": 1.0,
            "debug": debug,
        },
        name="AssistantRouter",
        daemon=True,
    )
    thread.start()
    return thread
