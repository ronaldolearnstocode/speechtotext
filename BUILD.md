# Building the standalone executable

This document describes how to build the **zip-distributable** Windows executable (one folder: exe + runtime + optional CUDA DLLs + config).

## Prerequisites

- **Windows** (the spec is for Windows; Linux/macOS would need a separate spec).
- **Python 3.10+** with dependencies installed:
  ```bash
  pip install -r requirements.txt
  pip install pyinstaller
  ```
- **Option B (bundle CUDA):** Install [CUDA 12 Toolkit, not 13 or above](https://developer.nvidia.com/cuda-downloads) on the build machine. Set **`CUDA_PATH`** to the toolkit root (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`), or leave it unset and the spec will try the default path. All `bin\*.dll` files from that folder are copied into the output so the zip works with GPU without requiring users to install CUDA.

## Build steps

1. From the **project root** (where `main.py` and `SpeechToText.spec` are):
   ```bash
   pyinstaller SpeechToText.spec
   ```

2. Output is in **`dist/SpeechToText/`**:
   - `SpeechToText.exe` — main executable
   - `config.yaml` — default config (users can edit next to the exe)
   - Many runtime DLLs and the Python/speechtotext stack
   - If you built with CUDA: CUDA 12 runtime DLLs (e.g. `cublas64_12.dll`) in the same folder

3. **Zip for distribution:**
   - Zip the entire **`dist/SpeechToText`** folder (e.g. `SpeechToText-v1.0.zip`).
   - Users unzip and run `SpeechToText.exe`; no install required.

## Notes

- **Whisper model** is not bundled; the first run will download it (e.g. large-v3) and cache it.
- **CUDA:** If `CUDA_PATH` was set (or default path found) at build time, the zip includes CUDA runtime DLLs and GPU works without installing CUDA on the target PC (NVIDIA driver still required).
- To build **without** bundling CUDA, clear `CUDA_PATH` and ensure no CUDA 12 at the default path; the spec will still build, and users can use `--cpu` or install CUDA themselves for `--gpu`.
- **PyInstaller 6+:** The spec uses `contents_directory='.'` so all files (config, DLLs) sit next to the exe. If the exe used to fail, rebuild with the current spec and try again.
- **If the exe misbehaves:** Run it from a command prompt (`cd dist\SpeechToText`, then `SpeechToText.exe` or `.\SpeechToText.exe`) so any error messages or tracebacks are visible.
