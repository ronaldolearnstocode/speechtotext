"""CUDA bin path for DLL search (stdlib only, no numpy/ctranslate2). Used before other imports so add_dll_directory runs first."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Current faster-whisper/ctranslate2 pip build expects CUDA 12 — prefer this
_CUDA_12_DLL = "cublas64_12.dll"
_CUDA_13_DLL = "cublas64_13.dll"


def get_cuda_bin_path() -> str | None:
    """
    Return a CUDA bin directory for DLL search (Windows), or None if not found.
    Prefers a path that contains cublas64_12.dll (CUDA 12), then cublas64_13.dll (CUDA 13).
    Uses CUDA_PATH, default toolkit path, then PATH.
    """
    if sys.platform != "win32":
        return None

    def bin_has_dll(bin_path: Path, dll: str) -> bool:
        return bin_path.is_dir() and (bin_path / dll).exists()

    def find_bin_prefer_12() -> str | None:
        """Return first bin that has cublas64_12.dll; else first that has cublas64_13.dll."""
        cuda_root = os.environ.get("CUDA_PATH")
        if cuda_root:
            bin_path = Path(cuda_root) / "bin"
            if bin_has_dll(bin_path, _CUDA_12_DLL):
                return str(bin_path)
            if bin_has_dll(bin_path, _CUDA_13_DLL):
                return str(bin_path)

        toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if toolkit.is_dir():
            v12_dirs = []
            v13_dirs = []
            for p in toolkit.iterdir():
                if not p.is_dir():
                    continue
                name = p.name.lower()
                bin_p = p / "bin"
                if name.startswith("v12.") or name == "v12" or name.startswith("12."):
                    if bin_has_dll(bin_p, _CUDA_12_DLL):
                        v12_dirs.append((p, name))
                elif name.startswith("v13.") or name == "v13" or name.startswith("13."):
                    if bin_has_dll(bin_p, _CUDA_13_DLL):
                        v13_dirs.append((p, name))
            v12_dirs.sort(key=lambda x: x[1], reverse=True)
            v13_dirs.sort(key=lambda x: x[1], reverse=True)
            for ver_dir, _ in v12_dirs:
                return str(ver_dir / "bin")
            for ver_dir, _ in v13_dirs:
                return str(ver_dir / "bin")

        for part in os.environ.get("PATH", "").split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if bin_has_dll(p, _CUDA_12_DLL):
                return str(p)
        for part in os.environ.get("PATH", "").split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if bin_has_dll(p, _CUDA_13_DLL):
                return str(p)
        return None

    return find_bin_prefer_12()
