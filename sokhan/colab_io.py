"""Google Colab I/O adapters.

Colab notebooks have no real audio/video devices on the Python side — the
mic, speaker and webcam all live in the *browser*. These classes are the
Colab equivalent of :class:`sokhan.audio_io.LocalAudio`: same
``start()``/``stop()`` shape, same effect (mic -> ``engine.feed_audio``,
engine audio -> speaker), just piped through a bit of JavaScript instead of
PortAudio. **The engine itself (``sokhan.engine``) is untouched** — this is
purely a transport adapter, exactly like ``LocalAudio`` is.

Requires a Colab (or Jupyter) environment with ``google.colab`` available.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("sokhan.colab_io")
SR = 16000

# ---------------------------------------------------------------------------
# Browser side: a persistent AudioWorklet-free recorder (ScriptProcessor is
# deprecated but universally supported and simplest for a notebook demo) that
# streams ~100 ms Float32 PCM chunks to Python, plus a small scheduler that
# plays back PCM chunks pushed from Python back-to-back with no gaps/clicks.
# ---------------------------------------------------------------------------
_JS_SETUP = r"""
(() => {
  if (window._sokhan) { return; }
  const S = window._sokhan = {
    ctxIn: null, ctxOut: null, stream: null, proc: null, src: null,
    playHead: 0, recording: false, level: 0,
    videoEl: null, camStream: null,
  };

  S.b64FromFloat32 = (f32) => {
    const bytes = new Uint8Array(f32.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 32768) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
    }
    return btoa(bin);
  };
  S.float32FromB64 = (b64) => {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  };

  // ---- microphone -> Python (16 kHz mono Float32, ~100 ms chunks)
  S.startMic = async () => {
    if (S.recording) return "already recording";
    S.stream = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
    S.ctxIn = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
    S.src = S.ctxIn.createMediaStreamSource(S.stream);
    S.proc = S.ctxIn.createScriptProcessor(2048, 1, 1);
    S.proc.onaudioprocess = (e) => {
      const inb = e.inputBuffer.getChannelData(0);
      let sum = 0; for (let i = 0; i < inb.length; i++) sum += inb[i]*inb[i];
      S.level = Math.sqrt(sum / inb.length);
      const copy = new Float32Array(inb);
      google.colab.kernel.invokeFunction('sokhan.on_audio_chunk', [S.b64FromFloat32(copy)], {});
    };
    S.src.connect(S.proc); S.proc.connect(S.ctxIn.destination);
    S.recording = true;
    return "mic started at " + S.ctxIn.sampleRate + " Hz";
  };
  S.stopMic = () => {
    if (S.proc) { S.proc.disconnect(); S.proc.onaudioprocess = null; }
    if (S.src) S.src.disconnect();
    if (S.stream) S.stream.getTracks().forEach(t => t.stop());
    if (S.ctxIn) S.ctxIn.close();
    S.recording = false;
    return "mic stopped";
  };
  S.micLevel = () => S.level;

  // ---- Python -> speaker (gapless scheduled playback, 24 kHz)
  S.initOut = (sr) => {
    if (!S.ctxOut) S.ctxOut = new (window.AudioContext || window.webkitAudioContext)({sampleRate: sr});
    S.playHead = S.ctxOut.currentTime;
    return "out ready";
  };
  S.playChunk = (b64, sr) => {
    if (!S.ctxOut) S.initOut(sr);
    const f32 = S.float32FromB64(b64);
    const buf = S.ctxOut.createBuffer(1, f32.length, sr);
    buf.copyToChannel(f32, 0);
    const node = S.ctxOut.createBufferSource();
    node.buffer = buf; node.connect(S.ctxOut.destination);
    const now = S.ctxOut.currentTime;
    const start = Math.max(now, S.playHead);
    node.start(start);
    S.playHead = start + buf.duration;
    return S.playHead - now;               // seconds still queued
  };
  S.flushOut = () => { if (S.ctxOut) S.playHead = S.ctxOut.currentTime; return "flushed"; };
  S.queuedSeconds = () => S.ctxOut ? Math.max(0, S.playHead - S.ctxOut.currentTime) : 0;

  // ---- webcam
  S.startCam = async () => {
    S.camStream = await navigator.mediaDevices.getUserMedia({video: {width: 320, height: 240}});
    S.videoEl = document.createElement('video');
    S.videoEl.srcObject = S.camStream; S.videoEl.muted = true; S.videoEl.playsInline = true;
    S.videoEl.style.cssText = "width:160px;border-radius:8px;position:fixed;top:8px;right:8px;z-index:9999";
    document.body.appendChild(S.videoEl);
    await S.videoEl.play();
    return "camera started";
  };
  S.stopCam = () => {
    if (S.camStream) S.camStream.getTracks().forEach(t => t.stop());
    if (S.videoEl && S.videoEl.parentNode) S.videoEl.parentNode.removeChild(S.videoEl);
    S.videoEl = null; S.camStream = null;
    return "camera stopped";
  };
  S.grabFrame = () => {
    if (!S.videoEl) return null;
    const c = document.createElement('canvas');
    c.width = S.videoEl.videoWidth || 320; c.height = S.videoEl.videoHeight || 240;
    c.getContext('2d').drawImage(S.videoEl, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.7);
  };
})();
"""


def _require_colab():
    try:
        from google.colab import output  # type: ignore
    except ImportError as e:
        raise RuntimeError("sokhan.colab_io requires a Google Colab runtime "
                           "(google.colab was not importable).") from e
    try:
        from IPython import get_ipython
        if get_ipython() is None or get_ipython().kernel is None:
            raise AttributeError
    except AttributeError:
        raise RuntimeError(
            "No live IPython/Colab kernel found. This means example_colab.py (or this call) was "
            "run as a plain script (e.g. `!python example_colab.py` or `python example_colab.py`), "
            "not executed inside actual notebook cells. eval_js()/register_callback() only work "
            "when the code runs *as a cell*, driven by the real Colab kernel.\n"
            "Fix: copy each '# %% [n]' section of example_colab.py into its own Colab cell and run "
            "the cells in order -- do not run the whole .py file at once. (Or: "
            "`!pip install -q jupytext && jupytext --to notebook example_colab.py`, then open the "
            "resulting .ipynb in Colab.)") from None
    return output


class ColabAudioIO:
    """Browser microphone -> engine.feed_audio ; engine audio -> browser speaker.

    Drop-in analogue of :class:`sokhan.audio_io.LocalAudio` for notebooks.
    Call :meth:`start` once (it injects the JS and asks for mic permission),
    :meth:`stop` to release the microphone.
    """

    def __init__(self, engine):
        self.engine = engine
        self._output = _require_colab()
        self._started = False
        self._unreg = None
        self._level = 0.0
        self._level_lock = threading.Lock()
        self._sr = engine.config.audio.output_sr

    def start(self) -> None:
        from IPython.display import Javascript, display  # type: ignore
        display(Javascript(_JS_SETUP))

        def on_chunk(b64: str) -> None:
            try:
                raw = base64.b64decode(b64)
                pcm = np.frombuffer(raw, dtype=np.float32)
                self.engine.feed_audio(pcm, SR)
            except Exception:
                log.exception("failed to decode mic chunk")

        self._output.register_callback("sokhan.on_audio_chunk", on_chunk)
        self._output.eval_js("_sokhan.startMic()")
        self._output.eval_js(f"_sokhan.initOut({self._sr})")
        self._unreg = self.engine.on("audio_out", self._on_engine_audio)
        self._started = True
        log.info("Colab mic/speaker started (grant the browser mic permission if prompted)")

    def _on_engine_audio(self, audio: np.ndarray, sr: int) -> None:
        if not self._started:
            return
        b64 = base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii")
        try:
            self._output.eval_js(f"_sokhan.playChunk('{b64}', {sr})", timeout_sec=5)
        except Exception:
            log.debug("playback push failed (tab backgrounded?)")

    def mic_level(self) -> float:
        """Approximate current input RMS (0..1), useful for a live level meter."""
        try:
            return float(self._output.eval_js("_sokhan.micLevel()"))
        except Exception:
            return 0.0

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._output.eval_js("_sokhan.stopMic()")
        except Exception:
            pass
        if self._unreg:
            self._unreg()
            self._unreg = None


class ColabCamera:
    """Optional webcam capture. Off unless explicitly started (matches the
    "camera off by default, can be enabled" requirement in the desktop app).

    Usage::

        cam = ColabCamera(engine)
        cam.start()                 # asks browser permission, shows a small preview
        cam.capture_and_attach()    # grabs the current frame for the *next* turn
        cam.start_auto(interval_s=4)  # or keep attaching a fresh frame periodically
    """

    def __init__(self, engine):
        self.engine = engine
        self._output = _require_colab()
        self._running = False
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()

    def start(self) -> None:
        from IPython.display import Javascript, display  # type: ignore
        display(Javascript(_JS_SETUP))
        self._output.eval_js("_sokhan.startCam()")
        self._running = True
        self.engine.config.vision.enabled = True

    def stop(self) -> None:
        self._auto_stop.set()
        if self._auto_thread:
            self._auto_thread.join(timeout=2)
        if self._running:
            try:
                self._output.eval_js("_sokhan.stopCam()")
            except Exception:
                pass
        self._running = False

    def capture_and_attach(self) -> bool:
        """Grab one frame from the webcam and attach it to the engine's next turn."""
        if not self._running:
            return False
        data_url = self._output.eval_js("_sokhan.grabFrame()")
        if not data_url:
            return False
        self.engine.set_image(data_url)
        return True

    def start_auto(self, interval_s: float = 4.0) -> None:
        """Keep attaching a fresh frame every ``interval_s`` seconds (background thread)."""
        if self._auto_thread and self._auto_thread.is_alive():
            return
        self._auto_stop.clear()

        def loop():
            while not self._auto_stop.wait(interval_s):
                try:
                    self.capture_and_attach()
                except Exception:
                    log.debug("auto frame capture failed")
        self._auto_thread = threading.Thread(target=loop, daemon=True, name="sokhan-colab-cam")
        self._auto_thread.start()

    def stop_auto(self) -> None:
        self._auto_stop.set()
