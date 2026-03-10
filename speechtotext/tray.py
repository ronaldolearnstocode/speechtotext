"""System tray icon and optional status window for Speech-to-Text."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
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


def _run_tk_window(
    root_ref: list,
    show_on_start: bool,
    output_queue: Queue | None = None,
    stop_event: threading.Event | None = None,
    always_on_top: bool = False,
    poll_ms: int = 120,
) -> None:
    """Run tkinter status window (with optional assistant answers area) in a dedicated thread."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Speech-to-Text")
    root.resizable(True, True)
    root.minsize(400, 300)
    root.geometry("920x600")
    root.attributes("-topmost", bool(always_on_top))
    root.withdraw()
    root.protocol("WM_DELETE_WINDOW", root.withdraw)

    header = ttk.Frame(root, padding=(10, 8))
    header.pack(side=tk.TOP, fill=tk.X)
    ttk.Label(header, text="Speech-to-Text – Assistant answers", font=("Segoe UI", 10)).pack(side=tk.LEFT)

    def minimize_to_tray() -> None:
        root.withdraw()

    ttk.Button(header, text="Minimize to tray", command=minimize_to_tray).pack(side=tk.RIGHT, padx=(8, 0))

    if output_queue is not None:
        body = ttk.Frame(root, padding=(10, 0, 10, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def _copy_all() -> None:
            txt = text.get("1.0", "end-1c")
            if not txt.strip():
                return
            root.clipboard_clear()
            root.clipboard_append(txt)

        def _clear() -> None:
            text.delete("1.0", tk.END)

        ttk.Button(header, text="Copy all", command=_copy_all).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(header, text="Clear", command=_clear).pack(side=tk.RIGHT)

        text = tk.Text(body, wrap=tk.WORD, undo=False)
        scroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _append_message(msg: dict) -> None:
            prompt = str(msg.get("prompt", "")).strip()
            response = str(msg.get("response", "")).strip()
            provider = str(msg.get("provider_used", msg.get("provider", ""))).strip() or "assistant"
            if not response:
                return
            stamp = time.strftime("%H:%M:%S")
            text.insert(tk.END, f"[{stamp}] {provider}\n")
            if prompt:
                text.insert(tk.END, f"Q: {prompt}\n")
            text.insert(tk.END, "A:\n")
            text.insert(tk.END, response + "\n")
            text.insert(tk.END, "-" * 72 + "\n\n")
            text.see(tk.END)
            root.deiconify()
            root.lift()

        def _poll() -> None:
            if stop_event is not None and stop_event.is_set():
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            if output_queue is not None:
                try:
                    while True:
                        item = output_queue.get_nowait()
                        if item is None:
                            continue
                        if isinstance(item, dict):
                            _append_message(item)
                except Empty:
                    pass
            root.after(max(30, int(poll_ms)), _poll)

        root.after(0, _poll)
    else:
        frame = tk.Frame(root, padx=24, pady=16)
        frame.pack()
        tk.Label(frame, text="Speech-to-Text is running", font=("Segoe UI", 12)).pack(pady=(0, 8))
        tk.Label(frame, text="Hold hotkey to record, release to transcribe.", font=("Segoe UI", 9)).pack(pady=(0, 12))
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
    assistant_output_queue: Queue | None = None,
    assistant_output_window_topmost: bool = False,
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
                args=(
                    root_ref,
                    start_visible,
                    assistant_output_queue,
                    stop_event,
                    assistant_output_window_topmost,
                ),
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
