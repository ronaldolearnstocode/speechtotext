# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Speech-to-Text: one-folder build, optional CUDA DLLs (Option B).
# Run from project root: pyinstaller SpeechToText.spec
# Output: dist/SpeechToText/ (zip this folder for distribution).

import os
from pathlib import Path

# Resolve paths relative to this spec file
spec_dir = os.path.dirname(os.path.abspath(SPEC))
config_yaml = os.path.join(spec_dir, 'config.yaml')
datas = [(config_yaml, '.')] if os.path.isfile(config_yaml) else []

# faster_whisper VAD model: exe looks for faster_whisper/assets/silero_vad_v6.onnx next to exe
try:
    import faster_whisper
    fw_assets = Path(faster_whisper.__file__).resolve().parent / 'assets'
    if fw_assets.is_dir():
        datas.append((str(fw_assets), 'faster_whisper/assets'))
except Exception:
    pass

# Option B: bundle CUDA 12 runtime DLLs so GPU works without installing CUDA.
# Set CUDA_PATH to your CUDA 12 toolkit root (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6)
# or leave unset to try the default toolkit path. DLLs are copied next to the exe.
cuda_binaries = []
cuda_root = os.environ.get('CUDA_PATH')
if cuda_root:
    cuda_bin = Path(cuda_root) / 'bin'
    if cuda_bin.is_dir():
        for dll in cuda_bin.glob('*.dll'):
            cuda_binaries.append((str(dll), '.'))
if not cuda_binaries:
    default_cuda = Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')
    if default_cuda.is_dir():
        for vdir in sorted(default_cuda.iterdir(), key=lambda p: p.name, reverse=True):
            if vdir.is_dir():
                b = vdir / 'bin'
                if b.is_dir() and (b / 'cublas64_12.dll').exists():
                    cuda_binaries = [(str(dll), '.') for dll in b.glob('*.dll')]
                    break
            if cuda_binaries:
                break

binaries = list(cuda_binaries)

# Hidden imports so PyInstaller includes all required modules
hiddenimports = [
    'speechtotext',
    'speechtotext.audio_capture',
    'speechtotext.config_loader',
    'speechtotext.cuda_path',
    'speechtotext.paths',
    'speechtotext.hotkey',
    'speechtotext.injector',
    'speechtotext.transcriber',
    'speechtotext.tray',
    'yaml',
    'PIL',
    'PIL._tkinter_finder',
    'pystray',
    'pystray._win32',
    'keyboard',
    'pyautogui',
    'pyperclip',
    'numpy',
    'faster_whisper',
    'ctranslate2',
    'webrtcvad',
    'pyaudio',
]

a = Analysis(
    [os.path.join(spec_dir, 'main.py')],
    pathex=[spec_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpeechToText',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.zipfiles,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SpeechToText',
)
