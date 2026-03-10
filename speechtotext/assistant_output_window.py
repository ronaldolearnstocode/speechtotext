"""Assistant output window: resizable, selectable text panel for AI answers."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue


def run_output_window(
    *,
    output_queue: Queue,
    stop_event: threading.Event,
    always_on_top: bool = False,
    poll_ms: int = 120,
) -> None:
    """Run a simple always-available window for browsing/copying AI responses."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Assistant Output")
    root.geometry("900x560")
    root.minsize(520, 280)
    root.attributes("-topmost", bool(always_on_top))
    root.withdraw()
    # Keep window process alive when user clicks X so future answers can reopen it.
    root.protocol("WM_DELETE_WINDOW", root.withdraw)

    header = ttk.Frame(root, padding=(10, 8))
    header.pack(side=tk.TOP, fill=tk.X)
    body = ttk.Frame(root, padding=(10, 0, 10, 10))
    body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    title_var = tk.StringVar(value="Assistant answers (select and copy what you want)")
    ttk.Label(header, textvariable=title_var).pack(side=tk.LEFT)

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
        if stop_event.is_set():
            try:
                root.destroy()
            except Exception:
                pass
            return
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
    root.mainloop()


def start_output_window_thread(
    *,
    output_queue: Queue,
    stop_event: threading.Event,
    always_on_top: bool = False,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_output_window,
        kwargs={
            "output_queue": output_queue,
            "stop_event": stop_event,
            "always_on_top": always_on_top,
            "poll_ms": 120,
        },
        name="AssistantOutputWindow",
        daemon=True,
    )
    thread.start()
    return thread
