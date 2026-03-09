"""Local text-to-speech output with interrupt support.

Primary: Piper (if configured and available).
Fallback: Windows System.Speech voice.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

_STATE_LOCK = threading.Lock()
_CURRENT_PROC: subprocess.Popen | None = None
_CURRENT_TMP_WAV: str | None = None


def _track_proc(proc: subprocess.Popen, tmp_wav: str | None = None) -> bool:
    global _CURRENT_PROC, _CURRENT_TMP_WAV
    with _STATE_LOCK:
        _CURRENT_PROC = proc
        _CURRENT_TMP_WAV = tmp_wav
    try:
        while True:
            code = proc.poll()
            if code is not None:
                return code == 0
            time.sleep(0.05)
    finally:
        with _STATE_LOCK:
            _CURRENT_PROC = None
            wav_to_delete = _CURRENT_TMP_WAV
            _CURRENT_TMP_WAV = None
        if wav_to_delete:
            try:
                os.remove(wav_to_delete)
            except Exception:
                pass


def is_speaking() -> bool:
    with _STATE_LOCK:
        proc = _CURRENT_PROC
    return proc is not None and proc.poll() is None


def stop_speaking() -> bool:
    """Stop current TTS playback if any. Returns True if something was stopped."""
    with _STATE_LOCK:
        proc = _CURRENT_PROC
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.terminate()
        return True
    except Exception:
        try:
            proc.kill()
            return True
        except Exception:
            return False


def _speak_windows(text: str, voice_hint: str = "male", rate: int = 0, volume: int = 100) -> bool:
    safe_text = text.replace("'", "''")
    voice_hint_safe = (voice_hint or "male").replace("'", "''")
    rate = max(-10, min(10, int(rate)))
    volume = max(0, min(100, int(volume)))
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$voice=($s.GetInstalledVoices() | Where-Object {{$_.VoiceInfo.Name -match '{voice_hint_safe}'}} | Select-Object -First 1); "
        "if($voice){$s.SelectVoice($voice.VoiceInfo.Name)}; "
        f"$s.Rate={rate}; $s.Volume={volume}; "
        f"$s.Speak('{safe_text}');"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        return _track_proc(proc, None)
    except Exception:
        return False


def play_ack_beep(frequency_hz: int = 1200, duration_ms: int = 80) -> bool:
    """Play a short confirmation beep (Windows only)."""
    try:
        import winsound
    except Exception:
        return False
    frequency = max(37, min(32767, int(frequency_hz)))
    duration = max(20, min(1000, int(duration_ms)))
    try:
        winsound.Beep(frequency, duration)
        return True
    except Exception:
        return False


def _play_wav_sync(wav_path: str) -> bool:
    safe_wav = wav_path.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System; "
        f"$p=New-Object System.Media.SoundPlayer('{safe_wav}'); "
        "$p.PlaySync();"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        return _track_proc(proc, wav_path)
    except Exception:
        return False


def _speak_piper(text: str, piper_path: str, piper_model: str) -> bool:
    if not piper_path or not piper_model:
        return False
    piper_exe = Path(piper_path)
    piper_voice = Path(piper_model)
    if not piper_exe.is_file() or not piper_voice.is_file():
        return False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name
    try:
        proc = subprocess.run(
            [str(piper_exe), "--model", str(piper_voice), "--output_file", wav_path],
            input=text,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if proc.returncode != 0 or not Path(wav_path).is_file():
            try:
                os.remove(wav_path)
            except Exception:
                pass
            return False
        return _play_wav_sync(wav_path)
    except Exception:
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return False


def speak_text(
    text: str,
    *,
    provider: str = "windows",
    voice: str = "male",
    rate: int = 0,
    volume: int = 100,
    piper_path: str = "",
    piper_model: str = "",
) -> bool:
    """Speak text using configured provider; fallback to Windows voice."""
    msg = (text or "").strip()
    if not msg:
        return False
    provider = (provider or "windows").lower()
    if provider == "piper":
        if _speak_piper(msg, piper_path=piper_path, piper_model=piper_model):
            return True
    return _speak_windows(msg, voice_hint=voice, rate=rate, volume=volume)
