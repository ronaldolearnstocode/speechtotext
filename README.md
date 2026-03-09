# Speech-to-Text (Local MVP)

A local, high-performance voice-to-text desktop utility. Hold a global hotkey to record from your microphone, release to transcribe with AI, and have the text typed into the currently focused window—all on your machine, no cloud.

## Features

- **Fully local**: Audio and transcription stay on your device; no data is sent to the cloud.
- **Global hotkey**: Hold **Ctrl+Windows** (configurable) to record; release to transcribe and inject text at the cursor.
- **Whisper Large-v3**: Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Int8 quantization for good accuracy on CPU.
- **Low latency**: Target under 2 seconds for a ~10-second sentence on a modern CPU.

## Requirements

- **OS**: Windows or Linux
- **Python**: 3.10 or newer
- **RAM**: ~4 GB for the model (large-v3); 8 GB+ recommended for comfort
- **Microphone**: Default system mic

## Install

1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/ronaldolearnstocode/speechtotext.git
   cd speechtotext
   ```

2. Create a virtual environment and install dependencies (recommended; avoids system Python permission issues):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows (Command Prompt)
   # On Windows PowerShell, if activation fails with "running scripts is disabled", run once:
   #   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   # then:  .\venv\Scripts\Activate.ps1
   # source venv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```
   If you prefer not to use a venv and get "Access is denied" when installing, use: `pip install -r requirements.txt --user`. If you then get `ModuleNotFoundError: No module named 'pkg_resources'`, install setuptools for system Python (run PowerShell as Administrator): `pip install setuptools`

3. **Windows + PyAudio**: If `pip install PyAudio` fails, try:
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```
   or use a [pre-built wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) for your Python version.

4. **Optional**: [FFmpeg](https://ffmpeg.org/) on your PATH can help faster-whisper with some formats; for raw PCM from the mic it is not required.

## Usage

Run the app (keep the terminal open while using):

```bash
python main.py
```

Or with a custom config file:

```bash
python main.py --config path/to/config.yaml
```

- **Hold** the hotkey (default: **Ctrl+Windows**) to record from your microphone.
- **Release** the hotkey to stop recording; the clip is transcribed and then typed into the window that has focus (e.g. editor, chat, form).
- **Ctrl+C** in the terminal to exit (or use the **quit hotkey**, default **Ctrl+Shift+Q**, or **Quit** from the tray menu).

By default the app shows a **system tray icon**. Right‑click it for **Show window** / **Hide window** or **Quit**. Use `--no-tray` to run without the tray (e.g. in a server or headless environment):

```bash
python main.py --no-tray
```

To see if the hotkey is detected, run with `--debug`:
```bash
python main.py --debug
```
Then hold/release the hotkey; you should see `[hotkey] RECORDING START` and `[hotkey] RECORDING STOP`. If you see nothing, run the terminal **as Administrator** (right‑click → Run as administrator), or change `hotkey` in `config.yaml` to `ctrl+alt` and try again.

Use **`--cpu`** or **`--gpu`** to force CPU (int8) or GPU/CUDA (float16) for this run; useful to compare latency without editing config. When either is set, per-transcription time is printed (e.g. `Transcription (cuda): 0.42 s`). For `--gpu`, **CUDA 12** (e.g. CUDA Toolkit 12.x with `cublas64_12.dll` on PATH) is required; if you see "cublas64_12.dll is not found", install [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads) and add its `bin` folder to PATH, or use `--cpu`.

```bash
python main.py --cpu
python main.py --gpu
```

Keep the target application focused so the text is inserted at the cursor.

### Local assistant wake word (Phase 3)

You can route speech to local AI (instead of typing) by starting with a wake word:

- `command ...` -> local Ollama provider (active)
- `mother ...` -> reserved for phase 4 cloud provider mapping (disabled by default)

Examples:

- `command write a SQL window function for running total`
- `command explain this Python traceback`

Behavior rules:

- Wake word must be at the **start** of the transcribed sentence.
- If the wake word appears in the middle, text is treated as normal dictation and typed.
- The assistant speaks responses through TTS (`assistant_tts_provider` in `config.yaml`).
- While AI is speaking, hold and release the hotkey (`Ctrl+Alt` by default); speech stops on release.

## Configuration

Edit `config.yaml` in the project root (or pass `--config` to another file):

| Option | Description | Default |
|--------|-------------|---------|
| `hotkey` | Global hotkey combo | `ctrl+win` |
| `model_name` | Whisper model | `large-v3` |
| `device` | `cpu` or `cuda` | `cpu` |
| `compute_type` | e.g. `int8` for CPU | `int8` |
| `sample_rate` | Mic sample rate (Hz) | `16000` |
| `chunk_duration_ms` | Audio frame size | `30` |
| `vad_aggressiveness` | 0–3, silence detection | `2` |
| `type_interval` | Delay between keystrokes (s) | `0.02` |
| `quit_hotkey` | Hotkey to exit the app | `ctrl+shift+q` |
| `show_window_on_start` | Show status window at startup (tray only) | `false` |
| `assistant_enabled` | Enable wake-word assistant routing | `true` |
| `assistant_wake_word_map` | Wake-word to provider map | `{command: ollama, mother: gemini}` |
| `assistant_provider_enabled` | Provider enable flags | `{ollama: true, gemini: false}` |
| `assistant_mode` | `work` (lighter) or `quality` (stronger) | `work` |
| `assistant_model_work` | Local model in work mode | `qwen2.5-coder:7b` |
| `assistant_model_quality` | Local model in quality mode | `qwen2.5-coder:14b` |
| `assistant_tts_provider` | `windows` or `piper` | `windows` |

Environment overrides: `STT_MODEL`, `STT_DEVICE` (optional).

## How it works

The app uses four threads:

1. **Hotkey**: Listens for the global key combo (press = start recording, release = stop and submit).
2. **Audio producer**: Captures microphone (PyAudio) at 16 kHz mono and buffers until you release the hotkey, then pushes the chunk to a queue.
3. **Transcriber**: Loads faster-whisper once; takes audio from the queue, transcribes (English), and puts the text on another queue.
4. **Injector**: Takes text from the queue and types it with pyautogui into the focused window.

## Troubleshooting

- **Not recording or pasting**: Run `python main.py --debug`. When you hold the hotkey you should see `[hotkey] RECORDING START`; when you release, `[hotkey] RECORDING STOP`. If you see nothing, the global hotkey is not being detected: **run the terminal as Administrator** (right‑click the terminal/PowerShell icon → Run as administrator, then `cd` to the project and `python main.py`). Alternatively, in `config.yaml` set `hotkey: "ctrl+alt"` and try again (Ctrl+Alt often works without admin).
- **Hotkey not working**: On Windows, global hotkeys often require running the app with elevated privileges (Run as administrator). On Linux, ensure the process has access to the input device.
- **No sound / mic not detected**: Check the default microphone in system settings and that no other app has exclusive access. Try another mic or port.
- **Model download**: The first run downloads the Whisper model (e.g. large-v3); this can take a few minutes and requires internet. After that it runs offline from cache.
- **PyAudio install fails**: See the Windows note under Install; on Linux you may need `portaudio19-dev` (e.g. `sudo apt install portaudio19-dev`).

### GPU (CUDA 12)

For **`--gpu`** on Windows you need the [CUDA 12 Toolkit](https://developer.nvidia.com/cuda-downloads) installed (e.g. for NVIDIA RTX 5070 or other supported GPUs). The app will look for CUDA DLLs (e.g. `cublas64_12.dll`) in:

- The folder given by the **`CUDA_PATH`** environment variable (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`) — the app uses `CUDA_PATH\bin`, or  
- The default path `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` (latest v12.x found).

**If you see "cublas64_12.dll is not found" or "[main] CUDA path not found":**

1. Install CUDA 12 Toolkit if you have not, and ensure its **`bin`** folder is on your system PATH, or set **`CUDA_PATH`** to the toolkit root (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`).
2. In a **new** terminal, run `where cublas64_12.dll`. If it prints a path, the shell can find the DLL; run the app from that same terminal (`python main.py --gpu`). If `where` finds nothing, add the CUDA 12 `bin` folder to PATH in System Environment Variables and restart the terminal (and Cursor if you launch from it).
3. If it still fails, use `--cpu` for this run; the app will print a short message when the CUDA path is not found so you can fix PATH or `CUDA_PATH`.

## Running the executable (zip distribution)

Pre-built or self-built standalone builds are distributed as a **zip**. No install: unzip and run.

1. **Unzip** the archive (e.g. `SpeechToText-v1.0.zip`) into a folder.
2. **Optional:** Edit `config.yaml` in that folder to change hotkey, model, device, etc.
3. **Run** `SpeechToText.exe` (double-click or from a terminal). The first run may download the Whisper model.
4. **CLI options** work as with `python main.py`: `--cpu`, `--gpu`, `--no-tray`, `--config path\to\config.yaml`, `--debug`.

**GPU (Option B build):** If the zip was built with CUDA runtime DLLs bundled, `--gpu` works without installing the CUDA Toolkit; an NVIDIA driver is still required. Otherwise, install [CUDA 12](https://developer.nvidia.com/cuda-downloads) and add its `bin` folder to PATH (or set `CUDA_PATH`) for GPU support.

To **build** the zip yourself, see [BUILD.md](BUILD.md).

## Phase 2 (future)

- **GPU**: Use NVIDIA GPU (e.g. CUDA) for sub-second inference.
- **Streaming**: Show text as you speak instead of after release.
- **LLM post-processing**: Optional grammar/punctuation correction via a local Ollama (or similar) model.

## License

MIT. See [LICENSE](LICENSE).
