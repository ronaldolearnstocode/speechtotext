"""Local AI clients (Phase 3: Ollama)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def ask_ollama(
    *,
    prompt: str,
    model: str,
    host: str = "http://127.0.0.1:11434",
    timeout_s: float = 60.0,
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> str:
    """Query local Ollama and return response text."""
    url = host.rstrip("/") + "/api/generate"
    system = (
        "You are a concise coding assistant. "
        "Answer directly with practical steps and short examples when helpful."
    )
    payload = {
        "model": model,
        "prompt": f"{system}\n\nUser: {prompt}\nAssistant:",
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {host}. Is Ollama running?") from e
    except TimeoutError as e:
        raise RuntimeError("Ollama request timed out.") from e
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    text = str(parsed.get("response", "")).strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text
