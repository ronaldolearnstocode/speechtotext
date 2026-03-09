"""Assistant worker: consumes provider-tagged queries and speaks responses."""

from __future__ import annotations

import threading
from queue import Empty, Queue
import re

from speechtotext.local_ai import ask_ollama
from speechtotext.tts import speak_text


_UNKNOWN_FALLBACK = "sorry, I don't have an answer for that."


def _normalize_response(text: str) -> str:
    msg = (text or "").strip()
    if not msg:
        return _UNKNOWN_FALLBACK
    low = msg.lower()
    unknown_markers = (
        "i don't know",
        "i do not know",
        "don't have enough",
        "do not have enough",
        "not sure",
        "cannot answer",
        "can't answer",
        "no access to",
        "don't have access",
        "do not have access",
        "unable to",
    )
    if any(marker in low for marker in unknown_markers):
        return _UNKNOWN_FALLBACK
    return msg


def _build_voice_summary(text: str, max_chars: int = 160) -> str:
    """Build a short spoken summary from a long response."""
    msg = (text or "").strip()
    if not msg:
        return _UNKNOWN_FALLBACK
    # Remove code blocks to avoid reading noisy syntax aloud.
    msg = re.sub(r"```[\s\S]*?```", " ", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    # Prefer first sentence-like chunk.
    for sep in (". ", "? ", "! ", "; "):
        idx = msg.find(sep)
        if 20 <= idx <= max_chars:
            return msg[: idx + 1].strip()
    if len(msg) <= max_chars:
        return msg
    cut = msg[:max_chars].rstrip()
    return cut + "..."


def run_assistant_worker(
    *,
    assistant_query_queue: Queue,
    stop_event: threading.Event,
    provider_enabled: dict,
    ollama_host: str,
    model_work: str,
    model_quality: str,
    mode: str,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    tts_provider: str,
    tts_voice: str,
    tts_rate: int,
    tts_volume: int,
    tts_piper_path: str,
    tts_piper_model: str,
    assistant_output_queue: Queue | None = None,
    voice_summary_only: bool = True,
    voice_summary_max_chars: int = 160,
    debug: bool = False,
    queue_timeout: float = 1.0,
) -> None:
    """Consume assistant requests and return spoken answers."""
    model = model_quality if mode == "quality" else model_work
    enabled = {str(k).lower(): bool(v) for k, v in (provider_enabled or {}).items()}

    while not stop_event.is_set():
        try:
            item = assistant_query_queue.get(timeout=queue_timeout)
        except Empty:
            continue
        if item is None:
            break
        if not isinstance(item, dict):
            continue

        provider = str(item.get("provider", "")).strip().lower()
        prompt = str(item.get("prompt", "")).strip()
        wake_word = str(item.get("wake_word", "")).strip().lower()

        if not prompt:
            # Wake word was heard but no prompt text followed; beep-only UX handles feedback.
            if debug:
                print("[assistant] wake word detected without prompt")
            continue

        if not enabled.get(provider, False):
            speak_text(
                f"Provider for wake word {wake_word} is disabled.",
                provider=tts_provider,
                voice=tts_voice,
                rate=tts_rate,
                volume=tts_volume,
                piper_path=tts_piper_path,
                piper_model=tts_piper_model,
            )
            continue

        try:
            if provider == "ollama":
                response = ask_ollama(
                    prompt=prompt,
                    model=model,
                    host=ollama_host,
                    timeout_s=timeout_s,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif provider == "gemini":
                response = "Gemini provider is reserved for phase four and is not enabled yet."
            else:
                response = _UNKNOWN_FALLBACK
        except Exception as e:
            if debug:
                print(f"[assistant] error={e}")
            response = _UNKNOWN_FALLBACK

        response = _normalize_response(response)
        if debug:
            print(f"[assistant] provider={provider} mode={mode} model={model}")
            print(f"[assistant] response={response}")

        if assistant_output_queue is not None:
            assistant_output_queue.put(
                {
                    "provider": provider,
                    "prompt": prompt,
                    "response": response,
                }
            )

        spoken_text = _build_voice_summary(response, max_chars=voice_summary_max_chars) if voice_summary_only else response
        speak_text(
            spoken_text,
            provider=tts_provider,
            voice=tts_voice,
            rate=tts_rate,
            volume=tts_volume,
            piper_path=tts_piper_path,
            piper_model=tts_piper_model,
        )


def start_assistant_thread(
    assistant_query_queue: Queue,
    stop_event: threading.Event,
    provider_enabled: dict,
    ollama_host: str,
    model_work: str,
    model_quality: str,
    mode: str,
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    tts_provider: str,
    tts_voice: str,
    tts_rate: int,
    tts_volume: int,
    tts_piper_path: str,
    tts_piper_model: str,
    assistant_output_queue: Queue | None = None,
    voice_summary_only: bool = True,
    voice_summary_max_chars: int = 160,
    debug: bool = False,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_assistant_worker,
        kwargs={
            "assistant_query_queue": assistant_query_queue,
            "stop_event": stop_event,
            "provider_enabled": provider_enabled,
            "ollama_host": ollama_host,
            "model_work": model_work,
            "model_quality": model_quality,
            "mode": mode,
            "timeout_s": timeout_s,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tts_provider": tts_provider,
            "tts_voice": tts_voice,
            "tts_rate": tts_rate,
            "tts_volume": tts_volume,
            "tts_piper_path": tts_piper_path,
            "tts_piper_model": tts_piper_model,
            "assistant_output_queue": assistant_output_queue,
            "voice_summary_only": voice_summary_only,
            "voice_summary_max_chars": voice_summary_max_chars,
            "debug": debug,
            "queue_timeout": 1.0,
        },
        name="AssistantWorker",
        daemon=True,
    )
    thread.start()
    return thread
