Project: Speech to Text (Phase 1 - CPU MVP)
🎯 Objective
Build a local, high-performance voice-to-text desktop utility that replicates the core functionality of Wispr Flow.
The app must run entirely on-device, intercept audio via a global hotkey, transcribe it using AI, and inject the text into the active window currently focused cursor position.

💻 Environment & Hardware (Phase 1)
OS: Windows / Linux (Local Execution)

CPU: Intel i5-13600K (14 Cores / 20 Threads)

RAM: 64GB DDR4 (Targeting Whisper Large-v3 model)

IDE: Cursor

🏗️ System Architecture (Multi-Threaded)
To prevent UI lag and ensure real-time feel, the app must use a Producer-Consumer pattern:

Thread A (UI/Hotkey): Listens for Global Hotkeys (Ctrl + Windows Key).

Thread B (Audio Producer): Captures microphone input using PyAudio. Implements VAD (Voice Activity Detection) and stops recording when keys are released.

Thread C (AI Consumer): Watches a queue for new audio chunks. Uses faster-whisper on CPU (Int8 quantization) to transcribe.

Thread D (Injector): Uses pyautogui or pynput to type the resulting string into the focused application.

🛠️ Tech Stack Requirements
Transcription Engine: faster-whisper (Implementation of OpenAI's Whisper).

Audio Handling: PyAudio + WebRTCVAD (for silence detection).

Hotkeys: keyboard or pynput.

Text Injection: pyautogui.

📝 Functional Requirements
Language: English.

Hot-loading: Load the large-v3 model into RAM once at startup.

Trigger: Hold hotkey to record, release to transcribe (or tap to start/stop).

Accuracy: English (B2 or higher).

Latency: Target < 2 seconds for a 10-second sentence on CPU.

🚀 Future Scope (Phase 2)
Transition from CPU to NVIDIA RTX 5070 (CUDA) for sub-second inference.

Streaming transcription (text appearing while speaking).

LLM post-processing (Grammar correction via local Ollama).