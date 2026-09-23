"""A polished desktop voice assistant (Tkinter, no extra UI dependencies).

    python examples/02_desktop_assistant.py

Talk hands-free, type, interrupt, clone your own voice in 5 seconds, and let
it see through the webcam when the model supports vision.
"""
from __future__ import annotations

import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from sokhan import Config, Omni, State  # noqa: E402
from sokhan.audio import record  # noqa: E402








LOCAL_LLM = r""   # local test model ("" = default)

# Language: picks the STT (speech recognition) model and the greeting. "fa" = Persian specialists,
# anything else ("en", "de", "ar", ...) = multilingual Whisper STT + a Piper voice for that language.
LANGUAGE = "fa"

# ====TTS: leave TTS_BACKEND = "" to use the language's default voice.
# TTS_BACKEND = "kitten"            # "" | "kitten" (English only) | "pocket_tts" | "sherpa_onnx"
# TTS_MODEL = r"C:\Users\userame\Downloads\kitten_tts_mini_v0_8.onnx"  # voices.npz beside it
# TTS_VOICE = "Luna"                # Kitten: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
# TTS_SPEED = 1.2                   # speaking speed: 0.8 = slower, 1.0 = normal, 1.2 = faster

# defult is persian
TTS_BACKEND = ""
TTS_MODEL = ""
TTS_VOICE = ""
TTS_SPEED = 1.2   









# ------------------------------------------------------------------ theme
BG, PANEL, CARD, LINE = "#0e1016", "#151823", "#1c2030", "#262b3b"
INK, MUTE, FAINT = "#e9ebf2", "#8b91a5", "#555b70"
ACCENT, TEAL, VIOLET, AMBER, RED = "#7c8cff", "#35d0ba", "#a78bfa", "#ffb454", "#ff6b81"
USER_BUBBLE = "#2b3160"
STATE_COLOR = {State.IDLE: "#59607a", State.LISTENING: TEAL, State.THINKING: ACCENT,
               State.SPEAKING: VIOLET, State.LOADING: FAINT}
STATE_TEXT = {State.IDLE: "Ready - just start talking", State.LISTENING: "Listening...",
              State.THINKING: "Thinking...", State.SPEAKING: "Speaking - talk to interrupt",
              State.LOADING: "Loading..."}


def is_rtl(text: str) -> bool:
    return any("\u0590" <= ch <= "\u08ff" for ch in text[:40])


_LTR = re.compile(r"^[A-Za-z0-9\u06f0-\u06f9\u0660-\u0669.,:/%+\-_@#]+$")
_EDGE = re.compile(r"^([^\w\u0600-\u06ff]*)(.*?)([^\w\u0600-\u06ff]*)$", re.S)


def draw_rtl(cv: tk.Canvas, text: str, right: int, top: int, maxw: int, font: tuple, fill: str):
    """Lay out a right-to-left paragraph token by token (toolkits' own bidi support varies
    by platform). Returns (item ids, width, height)."""
    f = tkfont.Font(font=font)
    space, lh = f.measure(" "), f.metrics("linespace")
    units = []                                   # (pieces drawn right-to-left, width)
    for word in text.split():
        lead, core, trail = _EDGE.match(word).groups()
        if _LTR.match(word):                     # numbers / Latin keep their own order
            if units and units[-1][2]:
                pieces, w, _ = units.pop()
                s = pieces[0] + " " + word
                units.append(([s], f.measure(s), True))
            else:
                units.append(([word], f.measure(word), True))
            continue
        pieces = [p for p in (lead[::-1], core, trail[::-1]) if p]
        units.append((pieces, sum(f.measure(p) for p in pieces), False))
    lines, cur, cw = [], [], 0
    for u in units:
        if cur and cw + space + u[1] > maxw:
            lines.append(cur)
            cur, cw = [], 0
        cw += (space if cur else 0) + u[1]
        cur.append(u)
    if cur:
        lines.append(cur)
    ids, width = [], 0
    for li, line in enumerate(lines):
        x, y = right, top + li * lh
        for pieces, _w, _ltr in line:
            for p in pieces:                         # logical order, placed right to left
                pw = f.measure(p)
                ids.append(cv.create_text(x - pw, y, text=p, font=font, fill=fill, anchor="nw"))
                x -= pw
            x -= space
        width = max(width, right - x - space)
    return ids, width, max(1, len(lines)) * lh


if sys.platform == "win32":                     # crisp text on high-DPI screens
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x + (y - x) * t):02x}" for x, y in zip(a, b))


def rounded(canvas: tk.Canvas, x1, y1, x2, y2, r=16, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2,
           x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


# ------------------------------------------------------------------ widgets
class Orb(tk.Canvas):
    """The assistant's 'face': breathes when idle, listens with your voice, spins while thinking."""

    def __init__(self, master, size=190):
        super().__init__(master, width=size, height=size, bg=BG, highlightthickness=0)
        self.size, self.state, self.level, self.t = size, State.LOADING, 0.0, 0.0
        self.color = STATE_COLOR[State.LOADING]
        self._tick()

    def _tick(self):
        self.t += 0.05
        target = STATE_COLOR.get(self.state, FAINT)
        self.color = mix(self.color, target, 0.15)
        self.level *= 0.88
        self.delete("all")
        c, s = self.size / 2, self.size
        breathe = 0.5 + 0.5 * math.sin(self.t * (3.2 if self.state == State.SPEAKING else 1.4))
        base = s * 0.25 + s * 0.08 * min(1.0, self.level * 3)
        if self.state in (State.SPEAKING, State.LISTENING):
            base += s * 0.015 * breathe
        for i in range(7, 0, -1):                          # soft glow
            r = base + i * s * 0.022 * (1 + 0.4 * breathe)
            self.create_oval(c - r, c - r, c + r, c + r, fill=mix(BG, self.color, 0.06 + 0.02 * (7 - i)), outline="")
        self.create_oval(c - base, c - base, c + base, c + base, fill=self.color, outline="")
        hl = base * 0.55                                    # highlight
        self.create_oval(c - hl * 0.9, c - base * 0.8, c + hl * 0.1, c - base * 0.8 + hl,
                         fill=mix(self.color, "#ffffff", 0.35), outline="")
        if self.state == State.THINKING:                   # orbiting arc
            r = base + s * 0.07
            self.create_arc(c - r, c - r, c + r, c + r, start=(self.t * 260) % 360, extent=90,
                            style="arc", outline=mix(self.color, "#ffffff", 0.4), width=3)
        self.after(33, self._tick)


class ChatView(tk.Frame):
    """Scrollable, canvas-drawn chat with rounded bubbles and live-updating text."""

    def __init__(self, master, font):
        super().__init__(master, bg=BG)
        self.font = font
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.items: list = []
        self.canvas.bind("<Configure>", lambda e: self._layout())
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

    def add(self, role: str, text: str) -> dict:
        item = {"role": role, "text": text, "ids": ()}
        self.items.append(item)
        self._layout(len(self.items) - 1)
        return item

    def set(self, item: dict, text: str) -> None:
        item["text"] = text
        self._layout(self.items.index(item))

    def remove(self, item: dict) -> None:
        if item in self.items:
            for i in item["ids"]:
                self.canvas.delete(i)
            self.items.remove(item)
            self._layout()

    def clear(self) -> None:
        self.canvas.delete("all")
        self.items.clear()

    def _layout(self, start: int = 0) -> None:
        cv, W = self.canvas, max(300, self.canvas.winfo_width())
        y = 14
        for k, it in enumerate(self.items):
            if k < start:
                y = it["bottom"] + 10
                continue
            for i in it["ids"]:
                cv.delete(i)
            role, text = it["role"], it["text"] or " "
            if role == "sys":
                tid = cv.create_text(W / 2, y, text=text, fill=MUTE, font=(self.font[0], 9), anchor="n",
                                     width=W - 60, justify="center")
                bb = cv.bbox(tid)
                it["ids"], it["bottom"] = (tid,), bb[3]
                y = bb[3] + 10
                continue
            maxw = int(W * 0.74)
            color = INK if role != "partial" else MUTE
            if is_rtl(text):
                probe_ids, tw, th = draw_rtl(cv, text, 0, 0, maxw, self.font, color)
                for i in probe_ids:
                    cv.delete(i)
                bw, bh = tw + 28, th + 20
                left = 16 if role == "assistant" else W - 16 - bw
                tids, _, _ = draw_rtl(cv, text, left + bw - 14, y + 10, maxw, self.font, color)
            else:
                tid = cv.create_text(0, 0, text=text, fill=color, font=self.font, anchor="nw", width=maxw)
                x1, y1, x2, y2 = cv.bbox(tid)
                bw, bh = (x2 - x1) + 28, (y2 - y1) + 20
                left = 16 if role == "assistant" else W - 16 - bw
                cv.coords(tid, left + 14, y + 10)
                tids = [tid]
            fill = {"assistant": CARD, "user": USER_BUBBLE, "partial": BG}[role]
            rid = rounded(cv, left, y, left + bw, y + bh, r=18, fill=fill,
                          outline=LINE if role != "user" else "", dash=(3, 3) if role == "partial" else None)
            cv.tag_lower(rid)
            it["ids"], it["bottom"] = (rid, *tids), y + bh
            y += bh + 10
        cv.configure(scrollregion=(0, 0, W, y + 10))
        cv.yview_moveto(1.0)


class Button(tk.Label):
    """Flat pill button with hover (Tk buttons can't be styled on every OS)."""

    def __init__(self, master, text, command, bg=CARD, fg=INK, **kw):
        super().__init__(master, text=text, bg=bg, fg=fg, padx=14, pady=8, cursor="hand2", **kw)
        self._bg, self._cmd, self.enabled = bg, command, True
        self.bind("<Button-1>", lambda e: self.enabled and self._cmd())
        self.bind("<Enter>", lambda e: self.enabled and self.configure(bg=mix(self._bg, "#ffffff", 0.12)))
        self.bind("<Leave>", lambda e: self.configure(bg=self._bg))

    def style(self, text=None, bg=None, enabled=None):
        if text is not None:
            self.configure(text=text)
        if bg is not None:
            self._bg = bg
            self.configure(bg=bg)
        if enabled is not None:
            self.enabled = enabled
            self.configure(fg=INK if enabled else FAINT, cursor="hand2" if enabled else "arrow")


# ------------------------------------------------------------------ app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sokhan")
        self.geometry("520x780")
        self.minsize(420, 600)
        self.configure(bg=BG)
        fams = set(tkfont.families())
        face = next((f for f in ("Vazirmatn", "Segoe UI Variable Text", "Segoe UI", "SF Pro Text",
                                 "Inter", "Noto Sans", "DejaVu Sans") if f in fams), "TkDefaultFont")
        self.f_body, self.f_small, self.f_title = (face, 12), (face, 10), (face, 20, "bold")
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.omni: Omni | None = None
        self.mic = None
        self.cam = None
        self.cur = None
        self.partial = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(30, self._pump)
        threading.Thread(target=self._boot, daemon=True).start()

    # -------------------------------------------------- layout
    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=22, pady=(18, 0))
        tk.Label(head, text="Sokhan", font=self.f_title, bg=BG, fg=INK).pack(side="left")
        self.badge = tk.Label(head, text="  offline · on-device  ", font=self.f_small, bg=PANEL, fg=MUTE)
        self.badge.pack(side="right", pady=6)

        self.orb = Orb(self)
        self.orb.pack(pady=(6, 0))
        self.status = tk.Label(self, text="Starting...", font=self.f_body, bg=BG, fg=MUTE)
        self.status.pack()
        self.progress = tk.Canvas(self, height=4, bg=PANEL, highlightthickness=0, width=260)
        self.progress.pack(pady=(8, 4))
        self._bar = self.progress.create_rectangle(0, 0, 0, 4, fill=ACCENT, width=0)

        self.chat = ChatView(self, self.f_body)
        self.chat.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x", padx=16, pady=(10, 6))
        self.entry = tk.Entry(bar, font=self.f_body, bg=PANEL, fg=INK, insertbackground=INK, relief="flat",
                              highlightthickness=0)
        self.entry.pack(side="left", fill="x", expand=True, padx=(14, 6), pady=10)
        self.entry.insert(0, "Type a message...")
        self.entry.configure(fg=FAINT)
        self.entry.bind("<FocusIn>", self._clear_hint)
        self.entry.bind("<Return>", self._send)
        Button(bar, "Send ➤", self._send, bg=ACCENT, font=self.f_small).pack(side="right", padx=6, pady=6)

        tools = tk.Frame(self, bg=BG)
        tools.pack(fill="x", padx=16, pady=(0, 6))
        self.b_mic = Button(tools, "🎙  Start talking", self._toggle_mic, bg=mix(BG, TEAL, 0.35), font=self.f_small)
        self.b_mic.pack(side="left")
        self.b_stop = Button(tools, "■ Stop", lambda: self.omni and self.omni.interrupt(), font=self.f_small)
        self.b_stop.pack(side="left", padx=6)
        self.b_voice = Button(tools, "🗣  Clone voice", self._clone_menu, font=self.f_small)
        self.b_voice.pack(side="left")
        self.b_cam = Button(tools, "📷", self._toggle_cam, font=self.f_small)
        self.b_cam.pack(side="left", padx=6)
        Button(tools, "↺ New chat", self._reset, font=self.f_small).pack(side="right")
        for b in (self.b_mic, self.b_stop, self.b_voice, self.b_cam):
            b.style(enabled=False)

        self.footer = tk.Label(self, text="", font=(self.f_small[0], 9), bg=BG, fg=FAINT)
        self.footer.pack(pady=(0, 10))

    # -------------------------------------------------- engine
    def _boot(self):
        cfg = Config.for_language(LANGUAGE)
        if LOCAL_LLM and os.path.exists(LOCAL_LLM):
            cfg.llm.model = LOCAL_LLM
        if TTS_BACKEND:
            cfg.tts.backend, cfg.tts.model, cfg.tts.voice = TTS_BACKEND, TTS_MODEL, TTS_VOICE
        cfg.tts.speed = TTS_SPEED
        cfg.prompt.name = "سخن" if LANGUAGE == "fa" else "Sokhan"
        cfg.prompt.greeting = ("سلام! من سخن هستم. در خدمتم." if LANGUAGE == "fa"
                               else "Hi! I'm Sokhan. How can I help?")
        # cfg.vision.enabled = True     # turn on with a vision model (Qwen3.5 + mmproj) for the camera
        omni = Omni(cfg)
        post = lambda kind: (lambda **kw: self.q.put((kind, kw)))
        for ev in ("state", "partial_transcript", "transcript", "response_delta", "response_done", "interrupted",
                   "metrics", "tool_call", "warning", "error", "load_progress", "ready"):
            omni.on(ev, post(ev))
        omni.on("level", lambda speech: self.q.put(("level", {"v": speech})))
        omni.on("audio", lambda audio: self.q.put(("level", {"v": float(np.sqrt(np.mean(audio ** 2))) * 4})))
        self.omni = omni
        try:
            omni.start()
        except Exception as e:
            self.q.put(("fatal", {"error": e}))

    def _pump(self):
        try:
            for _ in range(200):
                kind, kw = self.q.get_nowait()
                self._on(kind, kw)
        except queue.Empty:
            pass
        self.after(30, self._pump)

    def _on(self, kind: str, kw: dict):
        chat = self.chat
        if kind == "load_progress":
            self.status.configure(text=kw["message"] or kw["stage"])
            self.progress.coords(self._bar, 0, 0, 260 * kw["fraction"], 4)
        elif kind == "ready":
            self.progress.pack_forget()
            caps = self.omni.capabilities
            self.b_mic.style(enabled=True)
            self.b_stop.style(enabled=True)
            self.b_voice.style(enabled=caps["voice_cloning"])
            self.b_cam.style(enabled=caps["vision"])
            hw = self.omni.hw
            self.footer.configure(text=f"{hw.physical_cores} CPU cores · {'GPU' if self.omni.plan.use_gpu else 'CPU'}"
                                       f" · {'voice cloning ✓' if caps['voice_cloning'] else 'no voice cloning'}")
            self._toggle_mic()
        elif kind == "state":
            self.orb.state = kw["state"]
            self.status.configure(text=STATE_TEXT.get(kw["state"], ""), fg=MUTE)
        elif kind == "level":
            self.orb.level = max(self.orb.level, kw["v"])
        elif kind == "partial_transcript":
            if self.partial is None:
                self.partial = chat.add("partial", kw["text"])
            else:
                chat.set(self.partial, kw["text"])
        elif kind == "transcript":
            if self.partial is not None:
                chat.remove(self.partial)
                self.partial = None
            chat.add("user", kw["text"])
            self.cur = None
        elif kind == "response_delta":
            if self.cur is None:
                self.cur = chat.add("assistant", "")
            chat.set(self.cur, self.cur["text"] + kw["text"])
        elif kind == "response_done":
            if self.cur is None and kw["text"]:           # spoken verbatim (greeting)
                chat.add("assistant", kw["text"])
            self.cur = None
        elif kind == "interrupted":
            self.cur = None
            chat.add("sys", "you interrupted")
        elif kind == "tool_call":
            chat.add("sys", f"🔧 {kw['name']}({kw['arguments']})")
        elif kind == "metrics":
            self.footer.configure(text=f"⚡ {kw.get('latency_ms', 0) / 1000:.2f} s to first word · STT "
                                       f"{kw.get('stt_ms', 0):.0f} ms · LLM {kw.get('llm_ttft_ms', 0):.0f} ms · "
                                       f"voice {kw.get('tts_first_ms', 0):.0f} ms")
        elif kind == "warning":
            chat.add("sys", "⚠ " + kw["message"])
        elif kind in ("error", "fatal"):
            chat.add("sys", f"✗ {kw.get('stage', 'startup')}: {kw['error']}")
            if kind == "fatal":
                self.status.configure(text="Could not start - see the message above", fg=RED)

    # -------------------------------------------------- actions
    def _clear_hint(self, _=None):
        if self.entry.cget("fg") == FAINT:
            self.entry.delete(0, "end")
            self.entry.configure(fg=INK)

    def _send(self, _=None):
        text = self.entry.get().strip()
        if not text or self.entry.cget("fg") == FAINT or not (self.omni and self.omni.ready):
            return
        self.entry.delete(0, "end")
        self.omni.send_text(text)

    def _toggle_mic(self):
        if not (self.omni and self.omni.ready):
            return
        if self.mic is None:
            try:
                self.mic = self.omni.listen()
            except Exception as e:
                self.chat.add("sys", f"Microphone unavailable ({e}). You can still type.")
                return
            self.b_mic.style(text="●  Live", bg=mix(BG, TEAL, 0.6))
        else:
            self.mic.stop()
            self.mic = None
            self.b_mic.style(text="🎙  Start talking", bg=mix(BG, TEAL, 0.35))

    def _clone_menu(self):
        if not self.b_voice.enabled:
            return
        m = tk.Menu(self, tearoff=0, bg=CARD, fg=INK, activebackground=ACCENT, bd=0)
        m.add_command(label="Record my voice (5 s)", command=self._clone_record)
        m.add_command(label="Use a WAV file...", command=self._clone_file)
        m.add_separator()
        for v in self.omni.tts.voices():
            m.add_command(label=f"Built-in: {v}", command=lambda v=v: self._clone(v))
        m.tk_popup(self.b_voice.winfo_rootx(), self.b_voice.winfo_rooty() - 10)

    def _clone_record(self):
        def work():
            was_live = self.mic is not None
            if was_live:
                self.after(0, self._toggle_mic)
            for s in (3, 2, 1):
                self.after(0,lambda s=s: self.status.configure(text=f"Recording in {s}...", fg=AMBER))
                time.sleep(0.7)
            self.after(0, lambda: self.status.configure(text="● Recording - read any sentence for 5 s", fg=RED))
            audio = record(5.0, 24000)
            self._clone(audio, sr=24000)
            if was_live:
                self.after(0, self._toggle_mic)
        threading.Thread(target=work, daemon=True).start()

    def _clone_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3")])
        if path:
            self._clone(path)

    def _clone(self, sample, sr=None):
        def work():
            try:
                self.after(0, lambda: self.status.configure(text="Learning the voice...", fg=AMBER))
                self.omni.clone_voice(sample, sr)
                self.after(0, lambda: self.chat.add("sys", "✓ New voice ready"))
                self.omni.say("سلام! این صدای جدید منه." if self.omni.config.language == "fa" else
                              "Hi! This is my new voice.", remember=False)
            except Exception as e:
                msg = f"✗ voice cloning failed: {e}"
                self.after(0, lambda: self.chat.add("sys", msg))
        threading.Thread(target=work, daemon=True).start()

    def _toggle_cam(self):
        if not self.b_cam.enabled:
            return
        from sokhan.vision import Webcam
        if self.cam is None:
            try:
                self.cam = Webcam(self.omni).start()
                self.b_cam.style(text="📷 On", bg=mix(BG, ACCENT, 0.5))
            except Exception as e:
                self.chat.add("sys", f"✗ camera: {e}")
        else:
            self.cam.stop()
            self.cam = None
            self.b_cam.style(text="📷", bg=CARD)

    def _reset(self):
        if self.omni and self.omni.ready:
            self.omni.reset()
        self.chat.clear()
        self.cur = self.partial = None

    def _close(self):
        try:
            if self.cam:
                self.cam.stop()
            if self.omni:
                threading.Thread(target=self.omni.close, daemon=True).start()
        finally:
            self.after(150, self.destroy)


if __name__ == "__main__":
    App().mainloop()
