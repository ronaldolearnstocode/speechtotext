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


def create_icon_image(size: int = 64, ready: bool = True) -> "Image.Image":
    """Create tray icon: green circle when ready, red when loading. Mic-style with 'STT' dots."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    margin = size // 8
    if ready:
        fill, outline = (40, 180, 80), (30, 140, 60)   # green
    else:
        fill, outline = (200, 60, 60), (160, 40, 40)    # red
    d.ellipse([margin, margin, size - margin, size - margin], fill=fill, outline=outline)
    if size >= 32:
        try:
            cx, cy = size // 2, size // 2
            for dx, dy in [(-8, -6), (0, 0), (8, 6)]:
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
    device_state: dict | None = None,
    reload_event: threading.Event | None = None,
    menu_update_callback_ref: list | None = None,
    cuda_available: bool = False,
) -> "pystray.Icon":
    """Create tray icon with Show/Hide, Use CPU/GPU (radio), Quit. Red icon until ready, then green."""
    pystray = _pystray()
    root_ref: list = []
    tk_thread: threading.Thread | None = None
    tk_start_lock = threading.Lock()

    def ensure_tk_thread(start_visible: bool = False) -> None:
        nonlocal tk_thread
        with tk_start_lock:
            if tk_thread is not None and tk_thread.is_alive():
                return
            tk_thread = threading.Thread(
                target=_run_tk_window,
                args=(root_ref, start_visible),
                daemon=True,
            )
            tk_thread.start()

    if show_window_on_start:
        ensure_tk_thread(start_visible=True)

    def get_root():
        return root_ref[0] if root_ref else None

    def on_show_window(icon: "pystray.Icon") -> None:
        ensure_tk_thread(start_visible=True)
        root = get_root()
        if root is not None:
            try:
                root.after(0, root.deiconify)
            except Exception:
                pass

    def on_hide_window(icon: "pystray.Icon") -> None:
        ensure_tk_thread(start_visible=False)
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

    use_device_toggle = device_state is not None and reload_event is not None

    def on_use_cpu(icon: "pystray.Icon") -> None:
        if not use_device_toggle or not device_state:
            return
        device_state["device"] = "cpu"
        device_state["compute_type"] = "int8"
        reload_event.set()
        if menu_update_callback_ref and menu_update_callback_ref[0]:
            menu_update_callback_ref[0]()

    def on_use_gpu(icon: "pystray.Icon") -> None:
        if not use_device_toggle or not device_state:
            return
        device_state["device"] = "cuda"
        device_state["compute_type"] = "float16"
        reload_event.set()
        if menu_update_callback_ref and menu_update_callback_ref[0]:
            menu_update_callback_ref[0]()

    def is_gpu_checked(_: object) -> bool:
        return bool(device_state and device_state.get("device") == "cuda")

    def is_cpu_checked(_: object) -> bool:
        return bool(not device_state or device_state.get("device") != "cuda")

    menu_items = [
        pystray.MenuItem("Show window", on_show_window, default=True),
        pystray.MenuItem("Hide window", on_hide_window),
    ]
    if use_device_toggle:
        menu_items.append(pystray.Menu.SEPARATOR)
        menu_items.append(
            pystray.MenuItem("Use CPU", on_use_cpu, radio=True, checked=is_cpu_checked)
        )
        menu_items.append(
            pystray.MenuItem(
                "Use GPU",
                on_use_gpu,
                radio=True,
                checked=is_gpu_checked,
                enabled=cuda_available,
            )
        )
    menu_items.append(pystray.Menu.SEPARATOR)
    menu_items.append(pystray.MenuItem("Quit", on_quit))

    menu = pystray.Menu(*menu_items)
    # Start red (not ready); transcriber will set green when model is loaded
    icon = pystray.Icon(
        "speechtotext",
        create_icon_image(64, ready=False),
        "Speech-to-Text",
        menu,
    )
    return icon
