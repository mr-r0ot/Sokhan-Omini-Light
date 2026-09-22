#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""example_colab.py — سخن روی Google Colab.

این فایل برای اجرا **سلولی** در کولب نوشته شده (هر بخش '# %%' یک سلول است).
می‌توانی همین فایل را با Jupytext باز کنی، یا محتوای هر سلول را در یک سلول
جدید کولب کپی کنی.

هدف: حس یک مدل omni واقعی و realtime — صحبت پیوسته با میکروفون مرورگر،
شنیدن صدای مدل بی‌درنگ در همان مرورگر، و امکان روشن‌کردن وبکم برای اینکه
مدل تصویر لحظه‌ای را هم ببیند. **معماری sokhan (OmniEngine) هیچ تغییری
نکرده** — این فایل فقط میکروفون/بلندگو/وبکم مرورگر کولب را به همان
OmniEngine وصل می‌کند (دقیقاً هم‌ارز LocalAudio در دسکتاپ).

نکته سخت‌افزاری کولب: GPU رایگان کولب برای LLM اختیاری است (config زیر
`hardware.use_gpu=True` می‌گذارد و اگر Runtime > Change runtime type را روی
GPU گذاشته باشی خودش استفاده می‌کند؛ وگرنه روی CPU کولب هم اجرا می‌شود).
"""

# %% [1] نصب وابستگی‌ها (فقط بار اول در هر Runtime لازم است)
# !pip install -q sherpa-onnx llama-cpp-python onnxruntime soundfile sentencepiece \
#                 huggingface_hub psutil pillow
#
# اگر GPU روشن است (Runtime > Change runtime type > T4 GPU) برای llama-cpp-python
# نسخهٔ CUDA نصب کن تا LLM روی GPU برود (اختیاری، سرعت را بالا می‌برد):
# !CMAKE_ARGS="-DGGML_CUDA=on" pip install -q llama-cpp-python --force-reinstall --no-cache-dir

# %% [2] کلون یا آپلود پکیج sokhan
# اگر sokhan/ را در گیت‌هاب گذاشتی:
#   !git clone https://github.com/<you>/sokhan.git && %cd sokhan
# یا کافی‌ست پوشهٔ sokhan/ همین پروژه را در فایل‌های کولب آپلود/کپی کنی.
import sys
sys.path.insert(0, ".")

from sokhan import Config, OmniEngine, State, Tool          # noqa: E402
from sokhan.colab_io import ColabAudioIO, ColabCamera        # noqa: E402
from sokhan import hardware as hw                             # noqa: E402
from IPython.display import display, HTML                     # noqa: E402
import ipywidgets as widgets                                   # noqa: E402
import time                                                    # noqa: E402

# %% [3] یک سفارش ساده به‌عنوان مثال ابزار (tool) — دقیقاً مثل example.py دسکتاپ


class Order:
    def __init__(self):
        self.items: list[tuple[str, int]] = []

    def add(self, name: str, qty: int = 1) -> str:
        self.items.append((name, qty))
        return f"«{name}» با تعداد {qty} اضافه شد."

    def summary(self) -> str:
        return "، ".join(f"{n} ×{q}" for n, q in self.items) or "سفارشی ثبت نشده."


order = Order()


def add_item(name: str, qty: int = 1) -> str:
    """Add a food item with a quantity to the customer's order."""
    return order.add(name, qty)


def get_order_summary() -> str:
    """Read back everything added to the order so far."""
    return order.summary()


def describe_what_you_see() -> str:
    """No-op placeholder: the model already receives the latest camera frame as an
    image with every turn when the camera is on, so it can just describe it directly."""
    return "تصویر همین الان به شما پیوست شده."


tools = [
    Tool.from_function(add_item, "Add one food item and quantity to the order"),
    Tool.from_function(get_order_summary, "Get everything currently in the order"),
]

# %% [4] کانفیگ — پیش‌فرض‌های معماری دست‌نخورده، فقط طبق محیط کولب تنظیم شده
cfg = Config()
cfg.prompt.name = "سخن"
cfg.prompt.greeting = "سلام! من سخن هستم، دستیار صوتی شما. چی میل دارید سفارش بدید؟"
cfg.prompt.extra = ("You are running as a live demo in Google Colab. Keep replies short and "
                    "conversational. If an image is attached, react to what's actually in it.")

# GPU کولب را خودکار امتحان کن؛ اگر GPU نبود خودش روی CPU اجرا می‌شود (بدون خطا)
cfg.hardware.use_gpu = True
info = hw.detect_hardware()
print("سخت‌افزار کولب:", info.summary())

# اگر رم کولب کم بود (نسخهٔ رایگان معمولاً ~12 گیگ) از پریست lowmem استفاده کن:
# cfg = Config.preset("lowmem")

# %% [5] ساخت و بارگذاری موتور (بار اول: دانلود مدل‌ها، چند دقیقه طول می‌کشد)
engine = OmniEngine(cfg, tools=tools)

status_lbl = widgets.HTML("در حال آماده‌سازی…")
display(status_lbl)


def _on_progress(label, fraction):
    status_lbl.value = f"⏳ در حال بارگذاری <b>{label}</b>… {int(fraction*100)}٪"


def _on_ready():
    status_lbl.value = "✅ آماده است — سلول بعدی را اجرا کن تا میکروفون روشن شود."


def _on_state(state):
    icons = {State.IDLE: "🟢 آماده", State.LISTENING: "🎙️ در حال شنیدن…",
            State.THINKING: "🤔 در حال فکر کردن…", State.SPEAKING: "🔊 در حال صحبت…"}
    status_lbl.value = icons.get(state, str(state))


engine.on("load_progress", _on_progress)
engine.on("ready", _on_ready)
engine.on("state", _on_state)
engine.start(block=False)

# اجرای این سلول را ادامه بده؛ همین‌جا صبر می‌کنیم تا بارگذاری تمام شود
engine.wait_ready(timeout=1800)
print("موتور آماده است.")

# %% [6] نمایش گفتگو به‌صورت زنده (رونوشت کاربر + پاسخ مدل)
chat_log = widgets.HTML("<i>گفتگو اینجا نمایش داده می‌شود…</i>")
display(chat_log)
_chat_html = []


def _push(line: str) -> None:
    _chat_html.append(line)
    chat_log.value = "<div style='direction:rtl;font-family:Vazirmatn,Tahoma;line-height:1.9'>" \
                     + "".join(_chat_html[-40:]) + "</div>"


engine.on("transcript", lambda text, final: _push(f"<p>🧑 {text}</p>"))
engine.on("assistant_text", lambda text: _push(text))          # streamed inline
engine.on("assistant_done", lambda text: _push("<br/>"))
engine.on("tool_call", lambda name, arguments: _push(f"<p style='color:#888'>🔧 {name}({arguments})</p>"))
engine.on("interrupted", lambda reason, spoken: _push("<p style='color:#c0392b'>⤺ قطع شد</p>"))
engine.on("warning", lambda message: _push(f"<p style='color:#b8860b'>⚠ {message}</p>"))
engine.on("error", lambda where, error: _push(f"<p style='color:red'>✗ {where}: {error}</p>"))

# %% [7] روشن‌کردن میکروفون مرورگر — اینجاست که مکالمهٔ زندهٔ صوتی شروع می‌شود
# (کروم به شما اجازهٔ دسترسی به میکروفون را می‌پرسد؛ قبول کن)
audio_io = ColabAudioIO(engine)
audio_io.start()
print("میکروفون روشن شد. با مدل فارسی صحبت کن؛ اگر وسط جملهٔ او حرف بزنی، حرفش قطع می‌شود (barge-in).")

# یک نوار سطح صدای زنده (اختیاری، فقط برای حس بصری realtime بودن)
level_bar = widgets.FloatProgress(value=0, min=0, max=1, description="🎚️ سطح صدا")
display(level_bar)


def _level_loop():
    while audio_io._started:
        level_bar.value = min(1.0, audio_io.mic_level() * 6)
        time.sleep(0.1)


import threading  # noqa: E402
threading.Thread(target=_level_loop, daemon=True).start()

# %% [8] وبکم — اختیاری و پیش‌فرض خاموش، دقیقاً طبق خواستهٔ تو
# این سلول را فقط اگر می‌خواهی مدل تصویر لحظه‌ای ببیند اجرا کن.
ENABLE_CAMERA = False        # <- پارامتر: True کن تا وبکم روشن شود

camera = ColabCamera(engine)
if ENABLE_CAMERA:
    camera.start()                      # اجازهٔ دوربین را در مرورگر قبول کن
    camera.start_auto(interval_s=4.0)   # هر ۴ ثانیه یک فریم تازه به مکالمه پیوست می‌شود
    print("وبکم روشن شد؛ هر چند ثانیه یک‌بار مدل آخرین فریم را می‌بیند.")
else:
    print("وبکم خاموش است. برای روشن‌کردن: ENABLE_CAMERA = True و همین سلول را دوباره اجرا کن.")

# %% [9] جایگزین تایپی (اگر میکروفون کار نکرد یا خواستی متنی تست کنی)
text_box = widgets.Text(placeholder="یا اینجا تایپ کن و Enter بزن…")
display(text_box)


def _on_submit(w):
    txt = w.value.strip()
    w.value = ""
    if txt:
        engine.send_text(txt)


text_box.on_submit(_on_submit)

# %% [10] پایان جلسه — این سلول را در انتها اجرا کن تا میکروفون/وبکم/موتور بسته شوند
# audio_io.stop()
# camera.stop()
# engine.close()
