"""Audio plumbing.  The engine itself is I/O-agnostic (push PCM in, get PCM out);
``LocalAudio`` is the batteries-included microphone + speaker adapter
(sounddevice/PortAudio: Windows, macOS, Linux)."""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
from typing import Deque, Optional

import numpy as np

log = logging.getLogger("sokhan.audio")


def to_float32(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim > 1:
        a = a.mean(axis=1 if a.shape[0] > a.shape[1] else 0)
    if a.dtype == np.int16:
        return (a.astype(np.float32) / 32768.0)
    if a.dtype.kind in "iu":
        return a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return a.astype(np.float32, copy=False)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly  # best quality, optional
        g = math.gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class PlaybackBuffer:
    """Thread-safe FIFO feeding the output stream; also reports what is audible
    right now (used as the echo reference for barge-in)."""

    def __init__(self, sr: int):
        self.sr = sr
        self._q: Deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._queued = 0
        self._recent: Deque[tuple] = collections.deque(maxlen=12)   # (t, db)
        self._last_audible = 0.0

    def write(self, audio: np.ndarray, sr: int) -> None:
        a = resample(audio, sr, self.sr)
        with self._lock:
            self._q.append(a)
            self._queued += len(a)

    def flush(self) -> None:
        with self._lock:
            self._q.clear()
            self._queued = 0
        self._recent.clear()

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        got = 0
        with self._lock:
            while got < n and self._q:
                head = self._q[0]
                take = min(n - got, len(head))
                out[got:got + take] = head[:take]
                got += take
                if take == len(head):
                    self._q.popleft()
                else:
                    self._q[0] = head[take:]
            self._queued -= got
        if got:
            db = 20 * math.log10(float(np.sqrt(np.mean(out[:got] ** 2))) + 1e-9)
            self._recent.append((time.monotonic(), db))
            self._last_audible = time.monotonic()
        return out

    # -- introspection ---------------------------------------------------
    def buffered_ms(self) -> float:
        return 1000.0 * self._queued / self.sr

    def is_active(self) -> bool:
        return self._queued > 0 or (time.monotonic() - self._last_audible) < 0.12

    def level_db(self) -> Optional[float]:
        now = time.monotonic()
        vals = [db for t, db in list(self._recent) if now - t < 0.25]
        return max(vals) if vals else None


class LocalAudio:
    """Microphone -> engine.feed_audio, engine audio -> speakers."""

    def __init__(self, engine, input_device=None, output_device=None):
        self.engine = engine
        cfg = engine.config.audio
        self.in_dev = input_device if input_device is not None else cfg.input_device
        self.out_dev = output_device if output_device is not None else cfg.output_device
        self.in_sr, self.out_sr = cfg.input_sr, cfg.output_sr
        self.playback: Optional[PlaybackBuffer] = None
        self._in = self._out = None
        self._muted = False

    def start(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:  # OSError when PortAudio is missing
            raise RuntimeError("sounddevice/PortAudio not available. Install `sounddevice` "
                               "(Linux: `sudo apt install libportaudio2`).") from e
        # ---- output first so we know its rate
        out_sr = self.out_sr
        for sr in (out_sr, int(sd.query_devices(self.out_dev, "output")["default_samplerate"]), 48000, 44100):
            try:
                self.playback = PlaybackBuffer(sr)
                pb = self.playback

                def out_cb(outdata, frames, tinfo, status, _pb=pb):
                    outdata[:, 0] = _pb.read(frames)

                self._out = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", device=self.out_dev,
                                            callback=out_cb, latency=self.engine.config.audio.output_latency)
                self._out.start()
                break
            except Exception as e:
                log.debug("output at %s Hz failed: %s", sr, e)
                self._out = None
        if self._out is None:
            raise RuntimeError("Could not open an audio output device.")

        def in_cb(indata, frames, tinfo, status):
            if not self._muted:
                self.engine.feed_audio(indata[:, 0].copy(), self.in_sr)

        self._in = sd.InputStream(samplerate=self.in_sr, channels=1, dtype="float32", device=self.in_dev,
                                  blocksize=512, callback=in_cb)
        self._in.start()
        self.engine.attach_playback(self.playback)

    def mute(self, on: bool = True) -> None:
        self._muted = on

    def stop(self) -> None:
        for s in (self._in, self._out):
            try:
                if s is not None:
                    s.stop(); s.close()
            except Exception:
                pass
        self._in = self._out = None

    @staticmethod
    def list_devices() -> str:
        import sounddevice as sd  # type: ignore
        return str(sd.query_devices())
