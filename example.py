#!/usr/bin/env python3
"""example.py — a tiny, friendly desktop app built on top of the ``sokhan`` package.

This is what a *third-party user* who wants simplicity would write: import the
engine, hook up a microphone and speakers, and show a clean Tkinter window.

Run it with:
    python example.py

First run downloads the models (a few GB) and then everything works fully
offline. Everything here is optional to read — the whole app is one file.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, ".")  # so `python example.py` works from a fresh clone too

from sokhan import Config, OmniEngine, State, Tool  # noqa: E402
from sokhan.audio_io import LocalAudio  # noqa: E402
from sokhan import hardware as hw  # noqa: E402

APP_TITLE = "سخن — دستیار صوتی رستوران"
BG = "#faf7f2"
ACCENT = "#c0392b"
FONT = ("Vazirmatn", 13) if sys.platform != "win32" else ("Segoe UI", 11)
FONT_FA = ("Vazirmatn", 14)


# ---------------------------------------------------------------- a toy "order" tool
class Order:
    def __init__(self):
        self.items: list[tuple[str, int]] = []

    def add(self, name: str, qty: int = 1) -> str:
        self.items.append((name, qty))
        return f"«{name}» با تعداد {qty} به سفارش اضافه شد."

    def summary(self) -> str:
        if not self.items:
            return "سفارشی ثبت نشده."
        return "، ".join(f"{n} ×{q}" for n, q in self.items)

    def clear(self) -> None:
        self.items.clear()


def build_tools(order: Order) -> list[Tool]:
    def add_item(name: str, qty: int = 1) -> str:
        """Add a food item with a quantity to the customer's order."""
        return order.add(name, qty)

    def get_order_summary() -> str:
        """Read back everything added to the order so far."""
        return order.summary()

    def clear_order() -> str:
        """Remove everything from the order (the customer changed their mind)."""
        order.clear()
        return "سفارش خالی شد."

    return [Tool.from_function(add_item, "Add one food item and quantity to the order"),
            Tool.from_function(get_order_summary, "Get everything currently in the order"),
            Tool.from_function(clear_order, "Clear the whole order")]


# ---------------------------------------------------------------- UI
class SokhanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("560x640")
        self.configure(bg=BG)
        self.minsize(460, 520)

        self.order = Order()
        self.engine: OmniEngine | None = None
        self.audio: LocalAudio | None = None
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._vision_on = tk.BooleanVar(value=False)
        self._image_path: str | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._pump_events)
        threading.Thread(target=self._boot, daemon=True).start()

    # -------------------------------------------------- layout
    def _build_ui(self) -> None:
        top = tk.Frame(self, bg=BG); top.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(top, text="سخن", font=("Vazirmatn", 22, "bold"), bg=BG, fg=ACCENT).pack(side="right")
        tk.Label(top, text="دستیار صوتی سفارش‌گیری رستوران", font=FONT_FA, bg=BG, fg="#555").pack(side="right", padx=(0, 10))

        # status pill
        self.status_var = tk.StringVar(value="در حال آماده‌سازی…")
        self.status_lbl = tk.Label(self, textvariable=self.status_var, font=FONT_FA, bg="#eee", fg="#333",
                                   padx=14, pady=6)
        self.status_lbl.pack(fill="x", padx=16, pady=6)

        # transcript area
        mid = tk.Frame(self, bg=BG); mid.pack(fill="both", expand=True, padx=16, pady=4)
        self.chat = tk.Text(mid, wrap="word", font=FONT_FA, bg="white", relief="flat", padx=12, pady=10,
                            state="disabled")
        self.chat.tag_configure("user", justify="right", foreground="#1a1a1a", spacing1=6, spacing3=6)
        self.chat.tag_configure("assistant", justify="right", foreground=ACCENT, spacing1=6, spacing3=10)
        self.chat.tag_configure("sys", justify="center", foreground="#999", font=(FONT_FA[0], 10))
        sb = ttk.Scrollbar(mid, command=self.chat.yview); self.chat["yscrollcommand"] = sb.set
        self.chat.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

        # level meter
        self.level = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self.level.pack(fill="x", padx=16, pady=(2, 8))

        # text input row (typing works even before/without a microphone)
        row = tk.Frame(self, bg=BG); row.pack(fill="x", padx=16, pady=(0, 6))
        self.entry = tk.Entry(row, font=FONT_FA, justify="right")
        self.entry.pack(side="right", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", self._on_send_text)
        tk.Button(row, text="ارسال", command=self._on_send_text, font=FONT, bg=ACCENT, fg="white",
                 relief="flat", padx=12).pack(side="left", padx=(6, 0))

        # controls row
        ctl = tk.Frame(self, bg=BG); ctl.pack(fill="x", padx=16, pady=(0, 14))
        self.mic_btn = tk.Button(ctl, text="🎙 شروع مکالمه صوتی", command=self._toggle_mic, font=FONT,
                                 bg="#2e7d32", fg="white", relief="flat", padx=10, pady=6)
        self.mic_btn.pack(side="right")
        tk.Button(ctl, text="پاک‌کردن گفتگو", command=self._clear_chat, font=FONT).pack(side="right", padx=6)
        tk.Checkbutton(ctl, text="تصویر (دوربین/فایل) روشن", variable=self._vision_on, font=FONT, bg=BG,
                      command=self._toggle_vision).pack(side="left")
        tk.Button(ctl, text="پیوست تصویر…", command=self._attach_image, font=FONT).pack(side="left", padx=6)

        self.gpu_var = tk.StringVar(value="سخت‌افزار: در حال تشخیص…")
        tk.Label(self, textvariable=self.gpu_var, font=(FONT[0], 9), bg=BG, fg="#888").pack(pady=(0, 6))

    # -------------------------------------------------- engine boot
    def _boot(self) -> None:
        cfg = Config()                      # <- all sane defaults; edit here to customize
        cfg.prompt.name = "سخن"
        cfg.prompt.extra = ("You work for a Persian restaurant taking phone/voice orders. "
                            "Use the add_item / get_order_summary / clear_order tools to manage the cart. "
                            "Always read the final order back to the customer before saying it is placed.")
        cfg.prompt.greeting = "سلام! خوش اومدید. چی میل دارید سفارش بدید؟"
        # cfg.hardware.use_gpu = True        # <- user-configurable: uncomment to try GPU automatically
        info = hw.detect_hardware()
        gpu = f"GPU: {info.gpus[0].kind} {info.gpus[0].name}" if info.gpus else "GPU: none detected"
        self._events.put(("hw", f"CPU: {info.physical_cores} cores · RAM: {info.avail_ram_mb} MB free · {gpu}"))

        self.engine = OmniEngine(cfg, tools=build_tools(self.order))
        eng = self.engine
        eng.on("state", lambda state: self._events.put(("state", state)))
        eng.on("transcript", lambda text, final: self._events.put(("user_text", text)))
        eng.on("assistant_text", lambda text: self._events.put(("asst_delta", text)))
        eng.on("assistant_done", lambda text: self._events.put(("asst_done", text)))
        eng.on("interrupted", lambda reason, spoken: self._events.put(("sys", "⤺ قطع شد")))
        eng.on("level", lambda db, speech: self._events.put(("level", speech)))
        eng.on("warning", lambda message: self._events.put(("sys", "⚠ " + message)))
        eng.on("error", lambda where, error: self._events.put(("sys", f"✗ خطا در {where}: {error}")))
        eng.on("load_progress", lambda label, fraction: self._events.put(("progress", (label, fraction))))
        eng.on("ready", lambda: self._events.put(("ready", None)))
        eng.start(block=False)

    def _pump_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.after(40, self._pump_events)

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "hw":
            self.gpu_var.set(payload)
        elif kind == "progress":
            label, frac = payload
            self.status_var.set(f"در حال بارگذاری {label}… {int(frac*100)}٪")
        elif kind == "ready":
            self.status_var.set("آماده — دکمهٔ میکروفون را بزنید یا تایپ کنید")
            self.mic_btn.config(state="normal")
            self._append("سیستم آماده است.", "sys")
        elif kind == "state":
            names = {State.IDLE: "آماده", State.LISTENING: "در حال شنیدن…", State.THINKING: "در حال فکر کردن…",
                    State.SPEAKING: "در حال صحبت…", State.LOADING: "در حال بارگذاری…"}
            self.status_var.set(names.get(payload, str(payload)))
        elif kind == "user_text":
            self._append(payload, "user", prefix="🧑 ")
        elif kind == "asst_delta":
            self._append_stream(payload)
        elif kind == "asst_done":
            self._end_stream()
        elif kind == "sys":
            self._append(payload, "sys")
        elif kind == "level":
            self.level["value"] = min(100, payload * 100)

    # -------------------------------------------------- chat rendering
    _streaming = False

    def _append(self, text: str, tag: str, prefix: str = "") -> None:
        self.chat.config(state="normal")
        self.chat.insert("end", prefix + text + "\n", tag)
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _append_stream(self, delta: str) -> None:
        self.chat.config(state="normal")
        if not self._streaming:
            self.chat.insert("end", "🤖 ", "assistant")
            self._streaming = True
        self.chat.insert("end", delta, "assistant")
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _end_stream(self) -> None:
        if self._streaming:
            self.chat.config(state="normal")
            self.chat.insert("end", "\n")
            self.chat.config(state="disabled")
            self._streaming = False

    def _clear_chat(self) -> None:
        self.chat.config(state="normal"); self.chat.delete("1.0", "end"); self.chat.config(state="disabled")
        if self.engine:
            self.engine.reset_conversation()

    # -------------------------------------------------- actions
    def _on_send_text(self, *_):
        text = self.entry.get().strip()
        if not text or self.engine is None:
            return
        self.entry.delete(0, "end")
        image = self._image_path if self._vision_on.get() else None
        self.engine.send_text(text, image=image)

    def _toggle_mic(self) -> None:
        if self.engine is None:
            return
        if self.audio is None:
            try:
                self.audio = LocalAudio(self.engine)
                self.audio.start()
            except Exception as e:
                messagebox.showerror("میکروفون", f"راه‌اندازی صدا ممکن نشد:\n{e}\n\n"
                                     "می‌توانید بدون میکروفون هم با تایپ متن ادامه دهید.")
                self.audio = None
                return
            self.mic_btn.config(text="🛑 پایان مکالمه صوتی", bg=ACCENT)
        else:
            self.audio.stop()
            self.audio = None
            self.mic_btn.config(text="🎙 شروع مکالمه صوتی", bg="#2e7d32")
            self.level["value"] = 0

    def _toggle_vision(self) -> None:
        if self.engine:
            self.engine.config.vision.enabled = self._vision_on.get()
            if not self._vision_on.get():
                self.engine.clear_image()

    def _attach_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")])
        if path:
            self._image_path = path
            self._vision_on.set(True)
            self._toggle_vision()
            self._append(f"(تصویر پیوست شد: {path.split('/')[-1]})", "sys")

    def _on_close(self) -> None:
        try:
            if self.audio:
                self.audio.stop()
            if self.engine:
                self.engine.close()
        finally:
            self.destroy()


def main() -> None:
    app = SokhanApp()
    app.mic_btn.config(state="disabled")
    app.mainloop()


if __name__ == "__main__":
    main()
