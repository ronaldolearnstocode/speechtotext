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

Keep the target application focused so the text is inserted at the cursor.

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

## Phase 2 (future)

- **GPU**: Use NVIDIA GPU (e.g. CUDA) for sub-second inference.
- **Streaming**: Show text as you speak instead of after release.
- **LLM post-processing**: Optional grammar/punctuation correction via a local Ollama (or similar) model.

## License

MIT. See [LICENSE](LICENSE).
