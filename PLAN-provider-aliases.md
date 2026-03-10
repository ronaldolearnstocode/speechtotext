# Support `mother: "Cloud AI"` and `jarvis: "Local AI"` in configuration

## Goal

- **Mother (Cloud AI):** Wake word `mother: "Cloud AI"` (or `"gemini"`) runs the **cloud path**, which uses a **list** `cloud_ai_priority: [gemini, claude, groq]` — try each in order, first success wins. "Cloud AI" is a label; the list is normalized so "cloud ai" in config becomes "gemini" when iterating.
- **Jarvis (Local AI):** Wake word `jarvis: "Local AI"` runs the **local path**, which uses a **list** `local_ai_priority: [ollama]` — try each in order, first success wins. Today the list has only Ollama; later you can add other local backends. Both paths are list-driven: Cloud AI → `cloud_ai_priority`, Local AI → `local_ai_priority`.

## Data flow

- **config.yaml:** `mother: "Cloud AI"`, `jarvis: "Local AI"`, `cloud_ai_priority`, `local_ai_priority`
- **main.py:** Load both priority lists; normalize "cloud ai" to "gemini" in cloud list; merge provider_enabled (gemini/cloud ai, ollama/local ai); pass lists to worker
- **assistant_worker.py:** Provider "cloud ai" or "gemini" → loop `cloud_ai_priority`; provider "local ai" or "ollama" → loop `local_ai_priority`

## Config options (after implementation)

- **Mother (Cloud AI):** `assistant_wake_word_map: mother: "Cloud AI"` or `mother: "gemini"`; `assistant_provider_enabled: cloud ai: true` or `gemini: true`; `cloud_ai_priority: [Cloud AI, claude, groq]` or `[gemini, claude, groq]`; `assistant_cloud_ai_*` as aliases for `assistant_gemini_*`
- **Jarvis (Local AI):** `assistant_wake_word_map: jarvis: "Local AI"` or `jarvis: "ollama"`; `assistant_provider_enabled: local ai: true` or `ollama: true`; `local_ai_priority: [ollama]` — list of local backends

## Verification

- **Mother:** Config `mother: "Cloud AI"`, `cloud_ai_priority: [claude, groq]`. Say "Mother, &lt;question&gt;" and confirm Claude or Groq is used.
- **Jarvis:** Config `jarvis: "Local AI"`, `local_ai_priority: [ollama]`, `assistant_provider_enabled: local ai: true`. Say "Jarvis, &lt;question&gt;" and confirm Ollama is used via the list.
- Legacy: `jarvis: "ollama"` with no `local_ai_priority` still works (default list `[ollama]`).
