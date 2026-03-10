"""Quick test: call Groq API and print response or error. Use to verify GROQ_CLOUD_AI and connection."""
import os
import sys

# Run from repo root so speechtotext is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speechtotext.config_loader import load_config
from speechtotext.local_ai import ask_groq


def main():
    config = load_config()
    cloud_ai = str(os.environ.get("GROQ_CLOUD_AI") or "").strip()
    model = str(config.get("assistant_groq_model", "llama-3.3-70b-versatile")).strip()

    if not cloud_ai:
        print("No API key. Set GROQ_CLOUD_AI in environment (Groq uses GROQ, not GROK).")
        return 1

    print("Using key prefix:", cloud_ai[:8] + "..." if len(cloud_ai) > 8 else "(short)")
    print("Model:", model)
    print("Calling Groq...")
    try:
        r = ask_groq(prompt="Say hello in one short sentence.", model=model, api_key=cloud_ai)
        print("OK:", repr(r))
        return 0
    except Exception as e:
        print("Error:", type(e).__name__, e)
        if getattr(e, "__cause__", None):
            print("Cause:", e.__cause__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
