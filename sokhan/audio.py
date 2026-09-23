"""Audio plumbing.

The engine is transport-agnostic: push PCM in with ``feed_audio`` and take PCM
out from the ``audio`` event. ``LocalAudio`` is the batteries-included adapter
for a local microphone and speakers (sounddevice / PortAudio on Windows, macOS
and Linux). Anything else - a phone call, a WebSocket, a browser - is just
another adapter that implements the tiny ``AudioSink`` protocol.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
import wave
from typing import Deque, Optional, Protocol, Tuple

import numpy as np

log = logging.getLogger("sokhan.audio")


# --------------------------------------------------------------------------- conversions
def to_float32(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim > 1:                                      # (frames, channels) or (channels, frames)
        a = a.mean(axis=1 if a.shape[0] >= a.shape[1] else 0)
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    if a.dtype.kind in "iu":
        return a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return a.astype(np.float32, copy=False)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float32)))) + 1e-9)


def read_audio(path: str) -> Tuple[np.ndarray, int]:
    """Mono float32 + sample rate. WAV via stdlib; other formats need ``soundfile``."""
    try:
        import soundfile as sf  # type: ignore
        a, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_float32(a), int(sr)
    except ImportError:
        pass
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dtype).reshape(-1, ch)
    if sw == 1:
        a = (a.astype(np.float32) - 128) / 128.0
    return to_float32(a), sr


def write_wav(path: str, audio: np.ndarray, sr: int) -> str:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(to_pcm16(np.asarray(audio, np.float32)).tobytes())
    return path


# --------------------------------------------------------------------------- sink protocol
class AudioSink(Protocol):
    """What the engine needs from an output device (all methods thread-safe)."""

    def write(self, audio: np.ndarray, sr: int) -> None: ...
    def flush(self) -> None: ...                          # stop playing immediately
    def pending_seconds(self) -> float: ...               # audio queued but not yet heard
    def level_db(self) -> Optional[float]: ...            # loudness being played right now (echo ref)


class PlaybackBuffer:
    """Thread-safe FIFO between the engine and a device callback."""

    def __init__(self, sr: int):
        self.sr = sr
        self._q: Deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._queued = 0
        self._recent: Deque[Tuple[float, float]] = collections.deque(maxlen=32)

    def write(self, audio: np.ndarray, sr: int) -> None:
        a = resample(np.asarray(audio, np.float32), sr, self.sr)
        with self._lock:
            self._q.append(a)
            self._queued += len(a)

    def flush(self) -> None:
        with self._lock:
            self._q.clear()
            self._queued = 0

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
        self._recent.append((time.monotonic(), rms_db(out[:got]) if got else -120.0))
        return out

    def pending_seconds(self) -> float:
        return self._queued / self.sr

    def level_db(self) -> Optional[float]:
        now = time.monotonic()
        vals = [db for t, db in list(self._recent) if now - t < 0.2]
        return max(vals) if vals else None


# --------------------------------------------------------------------------- local device
class LocalAudio:
    """Microphone -> ``engine.feed_audio``; engine audio -> speakers."""

    def __init__(self, engine, input_device=None, output_device=None):
        self.engine = engine
        a = engine.config.audio
        self.in_dev = a.input_device if input_device is None else input_device
        self.out_dev = a.output_device if output_device is None else output_device
        self.playback: Optional[PlaybackBuffer] = None
        self._in = self._out = None
        self._muted = False

    def start(self) -> "LocalAudio":
        sd = _sounddevice()
        a = self.engine.config.audio
        native = a.output_sr or self.engine.output_sample_rate
        rates = [native, 48000, 44100]
        try:
            rates.insert(1, int(sd.query_devices(self.out_dev, "output")["default_samplerate"]))
        except Exception:
            pass
        for sr in dict.fromkeys(rates):                  # first rate the device accepts
            try:
                pb = PlaybackBuffer(sr)

                def out_cb(outdata, frames, tinfo, status, _pb=pb):
                    outdata[:, 0] = _pb.read(frames)

                self._out = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", device=self.out_dev,
                                            callback=out_cb, latency=a.output_latency)
                self._out.start()
                self.playback = pb
                break
            except Exception as e:
                log.debug("output at %s Hz failed: %s", sr, e)
                self._out = None
        if self._out is None:
            raise RuntimeError("could not open an audio output device")

        in_sr = a.input_sr
        block = max(160, int(in_sr * a.block_ms / 1000))

        watch = {"n": 0, "peak": 0.0}

        def in_cb(indata, frames, tinfo, status):
            if watch["n"] >= 0:                          # first ~3 s: is the microphone really live?
                watch["n"] += frames
                watch["peak"] = max(watch["peak"], float(np.abs(indata).max()))
                if watch["n"] >= 3 * in_sr:
                    if watch["peak"] < 1e-3:
                        msg = ("the microphone is sending silence - it is muted, its input volume is 0, "
                               "or the wrong device is selected (see sokhan.audio.list_devices())")
                        log.warning(msg)
                        self.engine._emit("warning", message=msg)
                    watch["n"] = -1
            if not self._muted:
                self.engine.feed_audio(indata[:, 0].copy(), in_sr)

        try:
            self._in = sd.InputStream(samplerate=in_sr, channels=1, dtype="float32", device=self.in_dev,
                                      blocksize=block, callback=in_cb, latency="low")
        except Exception:                                # device refuses 16 kHz: capture native, resample
            in_sr = int(sd.query_devices(self.in_dev, "input")["default_samplerate"])
            block = int(in_sr * a.block_ms / 1000)
            self._in = sd.InputStream(samplerate=in_sr, channels=1, dtype="float32", device=self.in_dev,
                                      blocksize=block, callback=in_cb, latency="low")
        self._in.start()
        self.engine.attach_sink(self.playback)
        return self

    def mute(self, on: bool = True) -> None:
        self._muted = on

    def stop(self) -> None:
        self.engine.detach_sink(self.playback)
        for s in (self._in, self._out):
            try:
                if s is not None:
                    s.stop()
                    s.close()
            except Exception:
                pass
        self._in = self._out = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def record(seconds: float = 5.0, sr: int = 24000, device=None) -> np.ndarray:
    """Record from the microphone (blocking). Handy for voice cloning."""
    sd = _sounddevice()
    a = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return a[:, 0].copy()


def list_devices() -> str:
    return str(_sounddevice().query_devices())


def _sounddevice():
    try:
        import sounddevice as sd  # type: ignore
        return sd
    except Exception as e:  # OSError when PortAudio is missing
        raise RuntimeError("audio I/O needs `pip install sounddevice` "
                           "(Linux: `sudo apt install libportaudio2`)") from e
