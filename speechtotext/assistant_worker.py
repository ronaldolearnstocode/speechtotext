"""Assistant worker: consumes provider-tagged queries and speaks responses."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from speechtotext.local_ai import ask_claude, ask_gemini, ask_groq, ask_ollama
from speechtotext.tts import speak_text

logger = logging.getLogger(__name__)


_UNKNOWN_FALLBACK = "sorry, I don't have an answer for that."


def _append_assistant_log(log_dir: str | Path, msg: dict) -> None:
    """Append one assistant Q/A to the hourly log file (YYYY-MM-DD-HH.log) in log_dir."""
    try:
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            return
        name = time.strftime("%Y-%m-%d-%H") + ".log"
        path = log_dir / name
        prompt = str(msg.get("prompt", "")).strip()
        response = str(msg.get("response", "")).strip()
        provider = str(msg.get("provider_used", msg.get("provider", ""))).strip() or "assistant"
        if not response:
            return
        stamp = time.strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {provider}\n")
            if prompt:
                f.write(f"Q: {prompt}\n")
            f.write("A:\n")
            f.write(response + "\n")
            f.write("-" * 72 + "\n\n")
            f.flush()
    except Exception:
        pass


def _append_speech_to_text_log(log_dir: str | Path, text: str) -> None:
    """Append one speech-to-text transcription to the hourly log file (YYYY-MM-DD-HH.log) in log_dir."""
    try:
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            return
        text = (text or "").strip()
        if not text:
            return
        name = time.strftime("%Y-%m-%d-%H") + ".log"
        path = log_dir / name
        stamp = time.strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] speech to text\n")
            f.write(f"Transcripted speech: {text}\n")
            f.write("-" * 72 + "\n\n")
            f.flush()
    except Exception:
        pass


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


def _build_voice_summary(text: str, max_chars: int = 280) -> str:
    """Build a short spoken summary: first sentence + concluding sentence when possible."""
    msg = (text or "").strip()
    if not msg:
        return _UNKNOWN_FALLBACK
    # Remove code blocks and markdown bullets so we don't read syntax aloud.
    msg = re.sub(r"```[\s\S]*?```", " ", msg)
    msg = re.sub(r"^\s*[\*\-]\s*", " ", msg, flags=re.MULTILINE)
    msg = re.sub(r"\*\*([^*]+)\*\*", r"\1", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    if len(msg) <= max_chars:
        return msg
    # Split into sentences (simple split on . ! ? followed by space).
    parts = re.split(r"(?<=[.!?])\s+", msg)
    sentences = [p.strip() for p in parts if (p and p.strip())]
    if not sentences:
        return msg[:max_chars].rstrip() + ("..." if len(msg) > max_chars else "")
    first = sentences[0]
    # If only one sentence or first is already a good standalone, return it capped.
    if len(sentences) == 1:
        return first[: max_chars] + ("..." if len(first) > max_chars else "")
    last = sentences[-1]
    # Prefer "first. In short, last." so the takeaway is spoken.
    if first == last:
        summary = first
    else:
        connector = " In short, " if len(last) < 200 else " "
        summary = first + connector + last
    if len(summary) <= max_chars:
        return summary
    # Trim to fit: keep first sentence, then " ... " and as much of last as fits.
    if len(first) + 5 <= max_chars:
        allowance = max_chars - len(first) - 5
        return first + " ... " + (last[:allowance].rstrip() + "..." if len(last) > allowance else last)
    return first[: max_chars].rstrip() + "..."


def _normalize_and_sentences(text: str) -> tuple[str, list[str]]:
    """Normalize response text and split into sentences. Returns (normalized_msg, sentences)."""
    msg = (text or "").strip()
    if not msg:
        return "", []
    msg = re.sub(r"```[\s\S]*?```", " ", msg)
    msg = re.sub(r"^\s*[\*\-]\s*", " ", msg, flags=re.MULTILINE)
    msg = re.sub(r"\*\*([^*]+)\*\*", r"\1", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    parts = re.split(r"(?<=[.!?])\s+", msg)
    sentences = [p.strip() for p in parts if (p and p.strip())]
    return msg, sentences


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
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    gemini_system_instruction: str = "",
    gemini_tts_provider: str = "piper",
    gemini_tts_voice: str = "Zira",
    gemini_tts_rate: int = 0,
    gemini_tts_volume: int = 100,
    gemini_tts_piper_path: str = "",
    gemini_tts_piper_model: str = "",
    cloud_ai_priority: list[str] | None = None,
    local_ai_priority: list[str] | None = None,
    claude_api_key: str = "",
    claude_model: str = "claude-3-5-haiku-20241022",
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    assistant_output_queue: Queue | None = None,
    assistant_log_dir: str | Path | None = None,
    voice_summary_only: bool = True,
    voice_summary_max_chars: int = 280,
    gemini_voice_summary_api_min_sentences: int = 5,
    debug: bool = False,
    queue_timeout: float = 1.0,
) -> None:
    """Consume assistant requests and return spoken answers. Local AI path uses local_ai_priority (e.g. ollama); Cloud AI path uses cloud_ai_priority (gemini, claude, groq)."""
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

        # Path enabled: Cloud AI (gemini/cloud ai), Local AI (ollama/local ai)
        is_enabled = (
            enabled.get(provider, False)
            or (provider == "cloud ai" and enabled.get("gemini", False))
            or (provider == "local ai" and enabled.get("ollama", False))
            or (provider == "ollama" and enabled.get("local ai", False))
        )
        if not is_enabled:
            logger.warning("Assistant provider disabled: wake_word=%s provider=%s", wake_word, provider)
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

        provider_used: str | None = None
        try:
            if provider in ("local ai", "ollama"):
                # Local AI path: try each backend in local_ai_priority (e.g. [ollama])
                response = None
                local_priority = [
                    s.strip().lower() for s in (local_ai_priority or ["ollama"]) if (s or "").strip()
                ]
                if not local_priority:
                    local_priority = ["ollama"]
                for name in local_priority:
                    if name == "ollama":
                        logger.info("Trying local provider: ollama")
                        try:
                            response = ask_ollama(
                                prompt=prompt,
                                model=model,
                                host=ollama_host,
                                timeout_s=timeout_s,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            provider_used = "ollama"
                            logger.info("Local provider succeeded: ollama")
                            break
                        except Exception as e:
                            logger.warning("Ollama failed: %s", e)
                            if debug:
                                print(f"[assistant] ollama failed: {e}")
                            continue
                if response is None:
                    response = _UNKNOWN_FALLBACK
            elif provider in ("cloud ai", "gemini"):
                response = None
                provider_used = None
                # Normalize "cloud ai" -> "gemini" in list so loop matches name == "gemini"
                priority = [
                    "gemini" if (s or "").strip().lower() == "cloud ai" else (s or "").strip().lower()
                    for s in (cloud_ai_priority or [])
                    if (s or "").strip()
                ]
                logger.info("assistant provider=cloud prompt=%r priority=%s", prompt[:80], priority)
                if not priority:
                    if (gemini_api_key or "").strip():
                        try:
                            response = ask_gemini(
                                prompt=prompt,
                                model=gemini_model,
                                api_key=gemini_api_key.strip(),
                                timeout_s=timeout_s,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                system_instruction=gemini_system_instruction,
                            )
                            provider_used = "gemini"
                        except Exception as e:
                            logger.exception("Gemini request failed")
                            if debug:
                                print(f"[assistant] error={e}")
                            response = f"Gemini error: {e}"
                    else:
                        logger.error("Gemini API key not set. Set GEMINI_CLOUD_AI in environment.")
                        response = "Gemini API key not set. Set GEMINI_CLOUD_AI in environment."
                    if response and not provider_used:
                        provider_used = "gemini"
                else:
                    # Try each cloud provider in order (Gemini → Claude → Groq); first success wins.
                    tried: list[str] = []
                    skipped: list[str] = []
                    last_error: str | None = None
                    logger.info("Cloud AI priority: %s", priority)
                    for name in priority:
                        if name == "gemini":
                            if not (gemini_api_key or "").strip():
                                skipped.append("gemini (no API key)")
                                logger.info("Skipping gemini: no API key")
                                continue
                            logger.info("Trying cloud provider: gemini")
                            tried.append("gemini")
                            try:
                                response = ask_gemini(
                                    prompt=prompt,
                                    model=gemini_model,
                                    api_key=gemini_api_key.strip(),
                                    timeout_s=timeout_s,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    system_instruction=gemini_system_instruction,
                                )
                                provider_used = "gemini"
                                logger.info("Cloud provider succeeded: gemini")
                                break
                            except Exception as e:
                                last_error = str(e)
                                logger.warning("Gemini failed: %s", e)
                                if debug:
                                    print(f"[assistant] gemini failed: {e}")
                                continue
                        elif name == "claude":
                            if not (claude_api_key or "").strip():
                                skipped.append("claude (no API key)")
                                logger.info("Skipping claude: no API key")
                                continue
                            logger.info("Trying cloud provider: claude")
                            tried.append("claude")
                            try:
                                response = ask_claude(
                                    prompt=prompt,
                                    model=claude_model,
                                    api_key=claude_api_key.strip(),
                                    timeout_s=timeout_s,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    system_instruction=gemini_system_instruction,
                                )
                                provider_used = "claude"
                                logger.info("Cloud provider succeeded: claude")
                                break
                            except Exception as e:
                                last_error = str(e)
                                logger.warning("Claude failed: %s", e)
                                if debug:
                                    print(f"[assistant] claude failed: {e}")
                                continue
                        elif name == "groq":
                            if not (groq_api_key or "").strip():
                                skipped.append("groq (no API key)")
                                logger.info("Skipping groq: no API key")
                                continue
                            logger.info("Trying cloud provider: groq")
                            tried.append("groq")
                            try:
                                response = ask_groq(
                                    prompt=prompt,
                                    model=groq_model,
                                    api_key=groq_api_key.strip(),
                                    timeout_s=timeout_s,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    system_instruction=gemini_system_instruction,
                                )
                                provider_used = "groq"
                                logger.info("Cloud provider succeeded: groq")
                                break
                            except Exception as e:
                                last_error = str(e)
                                logger.warning("Groq failed: %s", e)
                                if debug:
                                    print(f"[assistant] groq failed: {e}")
                                continue
                    if response is None:
                        logger.warning(
                            "No AI available. Tried: %s. Skipped: %s.",
                            tried or "(none)",
                            skipped or "(none)",
                        )
                        # Include tried/skipped and last error so the output window and log show what happened
                        parts = ["No AI available."]
                        if tried:
                            parts.append("Tried: " + ", ".join(tried) + ".")
                        if skipped:
                            parts.append("Skipped: " + ", ".join(skipped) + ".")
                        if last_error:
                            err_short = last_error[:120] + "..." if len(last_error) > 120 else last_error
                            parts.append("Last error: " + err_short)
                        response = " ".join(parts) if len(parts) > 1 else "No AI available."
                if response is not None:
                    response = (response or "").strip() or _UNKNOWN_FALLBACK
                else:
                    response = _UNKNOWN_FALLBACK
            else:
                response = _UNKNOWN_FALLBACK
        except Exception as e:
            logger.exception("Assistant request failed: provider=%s", provider)
            if debug:
                print(f"[assistant] error={e}")
            response = (
                _UNKNOWN_FALLBACK
                if provider not in ("gemini", "cloud ai")
                else f"Gemini error: {e}"
            )
            provider_used = None

        response = (response or "").strip() or _UNKNOWN_FALLBACK
        if debug:
            print(f"[assistant] provider={provider} mode={mode} model={model}")
            print(f"[assistant] response={response}")

        # Cloud path: log always shows real provider name (gemini, claude, groq) when one is used
        provider_used_val = provider_used if provider in ("gemini", "cloud ai") else provider
        if provider in ("gemini", "cloud ai") and provider_used_val is None:
            provider_used_val = "Cloud AI"
        output_msg = {"provider": provider, "provider_used": provider_used_val, "prompt": prompt, "response": response}
        if assistant_output_queue is not None:
            assistant_output_queue.put(output_msg)
        if assistant_log_dir:
            _append_assistant_log(assistant_log_dir, output_msg)

        if provider not in ("gemini", "cloud ai"):
            spoken_text = _build_voice_summary(response, max_chars=voice_summary_max_chars) if voice_summary_only else response
        elif provider in ("gemini", "cloud ai") and not voice_summary_only:
            spoken_text = response
        else:
            # Gemini + voice_summary_only: three-tier by sentence count
            msg, sentences = _normalize_and_sentences(response)
            n = len(sentences)
            if not msg:
                spoken_text = _UNKNOWN_FALLBACK
            elif n == 0:
                spoken_text = msg[:voice_summary_max_chars].rstrip() + ("..." if len(msg) > voice_summary_max_chars else "")
            elif n <= 2:
                spoken_text = sentences[0] if n == 1 else (sentences[0] + " " + sentences[1])
            elif n < gemini_voice_summary_api_min_sentences:
                first = sentences[0]
                last = sentences[-1]
                summary = first + " In short, " + last if first != last else first
                if len(summary) > voice_summary_max_chars:
                    allowance = voice_summary_max_chars - len(first) - 5
                    if allowance <= 0:
                        spoken_text = first[:voice_summary_max_chars].rstrip() + "..."
                    else:
                        spoken_text = first + " ... " + (last[:allowance].rstrip() + "..." if len(last) > allowance else last)
                else:
                    spoken_text = summary
            else:
                try:
                    summary_prompt = (
                        "Summarize the following in 1-2 full sentences for speaking aloud. "
                        "Capture the main point so the listener understands the gist. "
                        "Do not respond with only a few words or a name. Output only the summary.\n\n---\n"
                        + msg
                    )
                    spoken_text = ask_gemini(
                        prompt=summary_prompt,
                        model=gemini_model,
                        api_key=gemini_api_key.strip(),
                        timeout_s=timeout_s,
                        temperature=0.2,
                        max_tokens=100,
                        system_instruction="",
                    )
                    spoken_text = (spoken_text or "").strip()
                    if not spoken_text or len(spoken_text) < 40:
                        spoken_text = _build_voice_summary(response, max_chars=voice_summary_max_chars)
                except Exception:
                    spoken_text = _build_voice_summary(response, max_chars=voice_summary_max_chars)
        if provider in ("gemini", "cloud ai"):
            speak_text(
                spoken_text,
                provider=gemini_tts_provider,
                voice=gemini_tts_voice,
                rate=gemini_tts_rate,
                volume=gemini_tts_volume,
                piper_path=gemini_tts_piper_path,
                piper_model=gemini_tts_piper_model,
            )
        else:
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
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    gemini_system_instruction: str = "",
    gemini_tts_provider: str = "piper",
    gemini_tts_voice: str = "Zira",
    gemini_tts_rate: int = 0,
    gemini_tts_volume: int = 100,
    gemini_tts_piper_path: str = "",
    gemini_tts_piper_model: str = "",
    cloud_ai_priority: list[str] | None = None,
    local_ai_priority: list[str] | None = None,
    claude_api_key: str = "",
    claude_model: str = "claude-3-5-haiku-20241022",
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    assistant_output_queue: Queue | None = None,
    assistant_log_dir: str | Path | None = None,
    voice_summary_only: bool = True,
    voice_summary_max_chars: int = 280,
    gemini_voice_summary_api_min_sentences: int = 5,
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
            "gemini_api_key": gemini_api_key,
            "gemini_model": gemini_model,
            "gemini_system_instruction": gemini_system_instruction,
            "gemini_tts_provider": gemini_tts_provider,
            "gemini_tts_voice": gemini_tts_voice,
            "gemini_tts_rate": gemini_tts_rate,
            "gemini_tts_volume": gemini_tts_volume,
            "gemini_tts_piper_path": gemini_tts_piper_path,
            "gemini_tts_piper_model": gemini_tts_piper_model,
            "cloud_ai_priority": cloud_ai_priority,
            "local_ai_priority": local_ai_priority,
            "claude_api_key": claude_api_key,
            "claude_model": claude_model,
            "groq_api_key": groq_api_key,
            "groq_model": groq_model,
            "assistant_output_queue": assistant_output_queue,
            "assistant_log_dir": assistant_log_dir,
            "voice_summary_only": voice_summary_only,
            "voice_summary_max_chars": voice_summary_max_chars,
            "gemini_voice_summary_api_min_sentences": gemini_voice_summary_api_min_sentences,
            "debug": debug,
            "queue_timeout": 1.0,
        },
        name="AssistantWorker",
        daemon=True,
    )
    thread.start()
    return thread
