"""Local/cloud AI clients (Phase 3: Ollama, Gemini)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def ask_gemini(
    *,
    prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: str = "",
    timeout_s: float = 60.0,
    temperature: float = 0.3,
    max_tokens: int = 256,
    system_instruction: str = "",
) -> str:
    """Query Google Gemini API and return response text. Requires api_key from env GEMINI_CLOUD_AI."""
    if not (api_key or "").strip():
        raise RuntimeError("Gemini API key not set. Set GEMINI_CLOUD_AI in environment.")
    key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    if (system_instruction or "").strip():
        payload = {
            "contents": [{"parts": [{"text": f"User: {prompt}\nAssistant:"}]}],
            "systemInstruction": {"role": "system", "parts": [{"text": system_instruction.strip()}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
    else:
        system = (
            "You are a concise coding assistant. "
            "Answer directly with practical steps and short examples when helpful."
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\nUser: {prompt}\nAssistant:"}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
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
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            err_json = json.loads(err_body)
            msg = err_json.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        raise RuntimeError(f"Gemini API error: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Gemini API: {e}") from e
    except TimeoutError as e:
        raise RuntimeError("Gemini request timed out.") from e
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e

    candidates = parsed.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates. Check promptFeedback or try again.")
    c0 = candidates[0]
    parts = c0.get("content", {}).get("parts") or []
    if not parts:
        reason = c0.get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned empty content (finishReason: {reason}).")
    text = "".join(str(p.get("text", "")) for p in parts).strip()
    if not text:
        reason = c0.get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned an empty response (finishReason: {reason}).")
    return text


def ask_claude(
    *,
    prompt: str,
    model: str = "claude-3-5-haiku-20241022",
    api_key: str = "",
    timeout_s: float = 60.0,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    system_instruction: str = "",
) -> str:
    """Query Anthropic Claude Messages API and return response text. Requires api_key from env CLAUDE_CLOUD_AI."""
    if not (api_key or "").strip():
        raise RuntimeError("Claude API key not set. Set CLAUDE_CLOUD_AI in environment.")
    key = api_key.strip()
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if (system_instruction or "").strip():
        payload["system"] = system_instruction.strip()
    if 0 <= temperature <= 1:
        payload["temperature"] = temperature
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            err_json = json.loads(err_body)
            msg = err_json.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        raise RuntimeError(f"Claude API error: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Claude API: {e}") from e
    except TimeoutError as e:
        raise RuntimeError("Claude request timed out.") from e
    except Exception as e:
        raise RuntimeError(f"Claude request failed: {e}") from e

    content = parsed.get("content") or []
    text = "".join(
        str(block.get("text", ""))
        for block in content
        if block.get("type") == "text"
    ).strip()
    if not text:
        raise RuntimeError("Claude returned an empty response.")
    return text


def ask_groq(
    *,
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    api_key: str = "",
    timeout_s: float = 60.0,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    system_instruction: str = "",
) -> str:
    """Query Groq chat completions API and return response text. Requires api_key from env GROQ_CLOUD_AI."""
    if not (api_key or "").strip():
        raise RuntimeError("Groq API key not set. Set GROQ_CLOUD_AI in environment.")
    key = api_key.strip()
    url = "https://api.groq.com/openai/v1/chat/completions"
    messages = []
    if (system_instruction or "").strip():
        messages.append({"role": "system", "content": system_instruction.strip()})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "Groq-API-Client/1.0 (SpeechToText)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        msg = str(e)
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", {}).get("message", msg) or msg
        except Exception:
            pass
        if e.code == 403 and err_body:
            msg = msg + " | Raw: " + err_body[:500]
        raise RuntimeError(f"Groq API error: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Groq API: {e}") from e
    except TimeoutError as e:
        raise RuntimeError("Groq request timed out.") from e
    except Exception as e:
        raise RuntimeError(f"Groq request failed: {e}") from e

    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError("Groq returned no choices.")
    text = str(choices[0].get("message", {}).get("content", "")).strip()
    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return text


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
