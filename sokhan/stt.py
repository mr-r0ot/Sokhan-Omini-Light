"""Speech-to-text backends.

Default: Shenava (Persian FastConformer-CTC) through sherpa-onnx, CPU.
The model is an *offline* recognizer; at 30-100x realtime on a 32M model we
simply decode each VAD-segmented utterance (and, cheaply, growing partials).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import models
from .config import Config
from .text import itn_fa, normalize_chars

log = logging.getLogger("sokhan.stt")
SR = 16000


@dataclass
class STTResult:
    text: str
    raw: str = ""
    duration_s: float = 0.0
    decode_s: float = 0.0

    @property
    def rtf(self) -> float:
        return self.decode_s / self.duration_s if self.duration_s else 0.0


class STTBackend(ABC):
    @abstractmethod
    def load(self, plan=None, progress=None) -> None: ...
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int = SR) -> STTResult: ...
    def warmup(self) -> None:
        self.transcribe(np.zeros(SR, np.float32))
    def close(self) -> None: ...


class SherpaNemoCTC(STTBackend):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rec = None
        self.lock = threading.Lock()
        self._itn: Optional[Callable[[str], str]] = None

    def load(self, plan=None, progress=None) -> None:
        try:
            import sherpa_onnx  # type: ignore
        except ImportError as e:
            raise RuntimeError("sherpa-onnx is required for the default STT: `pip install sherpa-onnx`") from e
        d = models.ensure_stt(self.cfg, progress)
        threads = (plan.stt_threads if plan else self.cfg.stt.num_threads) or 2
        provider = plan.stt_provider if plan else "cpu"
        kw = dict(model=os.path.join(d, "model.onnx"), tokens=os.path.join(d, "tokens.txt"), num_threads=threads)
        try:
            self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(**kw, provider=provider)
        except TypeError:
            self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(**kw)
        # Model-specific ITN shipped in the repo (spoken numbers -> digits)
        self._itn = itn_fa
        p = os.path.join(d, "persian_itn.py")
        if os.path.exists(p):
            try:
                spec = importlib.util.spec_from_file_location("persian_itn", p)
                mod = importlib.util.module_from_spec(spec)           # type: ignore[arg-type]
                spec.loader.exec_module(mod)                           # type: ignore[union-attr]
                self._itn = mod.itn
            except Exception as e:
                log.warning("persian_itn.py failed to load (%s); using built-in ITN", e)

    def _prep(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from .audio_io import resample, to_float32
        x = resample(to_float32(audio), sr, SR)
        c = self.cfg.stt
        if c.auto_gain and len(x):
            r = float(np.sqrt(np.mean(x * x) + 1e-12))
            db = 20 * np.log10(r + 1e-9)
            if db < -45:                                   # basically silence: leave it alone
                pass
            else:
                gain = float(np.clip(10 ** ((c.target_dbfs - db) / 20), 0.5, 10.0))
                x = np.clip(x * gain, -1.0, 1.0)
        pad = np.zeros(int(0.15 * SR), np.float32)          # context for the first/last phoneme
        return np.concatenate([pad, x, pad]).astype(np.float32)

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> STTResult:
        assert self.rec is not None, "STT not loaded"
        dur = len(audio) / sr
        if dur * 1000 < self.cfg.stt.min_audio_ms:
            return STTResult("", "", dur, 0.0)
        x = self._prep(audio, sr)
        t = time.perf_counter()
        with self.lock:
            s = self.rec.create_stream()
            s.accept_waveform(SR, x)
            self.rec.decode_stream(s)
            raw = s.result.text
        dt = time.perf_counter() - t
        text = normalize_chars(raw).strip()
        if self.cfg.stt.itn and self._itn:
            try:
                text = self._itn(text)
            except Exception:
                pass
        return STTResult(text, raw, dur, dt)

    def try_transcribe(self, audio: np.ndarray, sr: int = SR) -> Optional[STTResult]:
        """Non-blocking variant for partials: returns None if the recognizer is busy."""
        if not self.lock.acquire(blocking=False):
            return None
        self.lock.release()
        return self.transcribe(audio, sr)


class MockSTT(STTBackend):
    """Deterministic STT for tests/demos: returns ``script[i]`` or ``fn(audio)``."""

    def __init__(self, script=None, fn: Optional[Callable[[np.ndarray], str]] = None, delay: float = 0.01):
        self.script, self.fn, self.delay, self.i = list(script or []), fn, delay, 0
        self.calls = 0

    def load(self, plan=None, progress=None) -> None: ...

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> STTResult:
        self.calls += 1
        time.sleep(self.delay)
        if self.fn:
            text = self.fn(audio)
        elif self.script:
            text = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
        else:
            text = "سلام"
        return STTResult(text, text, len(audio) / sr, self.delay)

    def try_transcribe(self, audio: np.ndarray, sr: int = SR) -> Optional[STTResult]:
        if self.fn:
            return self.transcribe(audio, sr)
        return STTResult("...", "...", len(audio) / sr, 0.0)


def create_stt(cfg: Config) -> STTBackend:
    if cfg.stt.backend == "mock":
        return MockSTT()
    return SherpaNemoCTC(cfg)
