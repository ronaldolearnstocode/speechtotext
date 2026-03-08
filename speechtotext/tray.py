"""System tray icon and optional status window for Speech-to-Text."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import pystray

# Lazy import pystray so we can run without tray (--no-tray)
_icon_module = None

def _pystray():
    global _icon_module
    if _icon_module is None:
        import pystray
        _icon_module = pystray
    return _icon_module


def create_icon_image(size: int = 64) -> "Image.Image":
    """Create a simple tray icon (mic-style circle with 'STT') using Pillow."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Circle background (blue)
    margin = size // 8
    d.ellipse([margin, margin, size - margin, size - margin], fill=(60, 120, 200), outline=(40, 90, 160))
    # Small "STT" text (white) - simplified as a dot/bar for very small sizes
    if size >= 32:
        try:
            # Draw "STT" with default font; fallback to minimal shape if no font
            cx, cy = size // 2, size // 2
            for i, (dx, dy) in enumerate([(-8, -6), (0, 0), (8, 6)]):
                d.ellipse([cx + dx - 4, cy + dy - 4, cx + dx + 4, cy + dy + 4], fill=(255, 255, 255))
        except Exception:
            d.ellipse([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(255, 255, 255))
    return img


def _run_tk_window(root_ref: list, show_on_start: bool) -> None:
    """Run tkinter status window in a dedicated thread (create + mainloop)."""
    import tkinter as tk
    root = tk.Tk()
    root.title("Speech-to-Text")
    root.resizable(False, False)
    root.withdraw()
    frame = tk.Frame(root, padx=24, pady=16)
    frame.pack()
    tk.Label(frame, text="Speech-to-Text is running", font=("Segoe UI", 12)).pack(pady=(0, 8))
    tk.Label(frame, text="Hold hotkey to record, release to transcribe.", font=("Segoe UI", 9)).pack(pady=(0, 12))
    def minimize_to_tray() -> None:
        root.withdraw()
    tk.Button(frame, text="Minimize to tray", command=minimize_to_tray).pack()
    root_ref.append(root)
    if show_on_start:
        root.after(0, root.deiconify)
    root.mainloop()


def create_tray_icon(
    stop_event: threading.Event,
    show_window_on_start: bool = False,
) -> "pystray.Icon":
    """Create a pystray icon with menu (Show window, Hide window, Quit). Optional tkinter window."""
    pystray = _pystray()
    root_ref: list = []  # [tk.Tk] when window is used
    tk_thread: threading.Thread | None = None

    if show_window_on_start:
        tk_thread = threading.Thread(target=_run_tk_window, args=(root_ref, True), daemon=True)
        tk_thread.start()
    else:
        # Still create the window so "Show window" works; start hidden
        tk_thread = threading.Thread(target=_run_tk_window, args=(root_ref, False), daemon=True)
        tk_thread.start()

    def get_root():
        return root_ref[0] if root_ref else None

    def on_show_window(icon: "pystray.Icon") -> None:
        root = get_root()
        if root is not None:
            try:
                root.after(0, root.deiconify)
            except Exception:
                pass

    def on_hide_window(icon: "pystray.Icon") -> None:
        root = get_root()
        if root is not None:
            try:
                root.after(0, root.withdraw)
            except Exception:
                pass

    def on_quit(icon: "pystray.Icon") -> None:
        stop_event.set()
        root = get_root()
        if root is not None:
            try:
                root.after(0, root.quit)
            except Exception:
                pass
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Show window", on_show_window, default=True),
        pystray.MenuItem("Hide window", on_hide_window),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon(
        "speechtotext",
        create_icon_image(64),
        "Speech-to-Text",
        menu,
    )
    return icon
