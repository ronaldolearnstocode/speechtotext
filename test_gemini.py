"""Quick test: call Gemini and print response or error."""
import os
import sys

# run from repo root so speechtotext is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speechtotext.config_loader import load_config
from speechtotext.local_ai import ask_gemini

def main():
    config = load_config()
    cloud_ai = str(os.environ.get("GEMINI_CLOUD_AI") or "").strip()
    model = os.environ.get("GEMINI_MODEL") or config.get("assistant_gemini_model", "gemini-2.5-flash")
    if not cloud_ai:
        print("No API key. Set GEMINI_CLOUD_AI in environment.")
        return 1
    print("Calling Gemini...")
    try:
        r = ask_gemini(prompt="Say hello in one short sentence.", model=model, api_key=cloud_ai)
        print("OK:", repr(r))
        return 0
    except Exception as e:
        print("Error:", type(e).__name__, e)
        return 1

if __name__ == "__main__":
    sys.exit(main())