"""Text-to-speech backends + phrase cache.

Default: pocket-tts-farsi-v2 through the pure-ONNX engine from
github.com/nimaone/persian_tts (CPU only, 24 kHz, voice cloning from a <=5 s wav).
"""
from __future__ import annotations

import collections
import hashlib
import inspect
import logging
import math
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from . import models
from .config import Config

log = logging.getLogger("sokhan.tts")


class TTSBackend(ABC):
    sample_rate: int = 24000

    @abstractmethod
    def load(self, plan=None, progress=None) -> None: ...
    @abstractmethod
    def synth(self, text: str) -> np.ndarray: ...
    def warmup(self) -> None:
        self.synth("سلام")
    def close(self) -> None: ...


def _as_audio(a) -> np.ndarray:
    if isinstance(a, tuple):
        a = a[0]
    a = np.asarray(a).squeeze()
    if a.dtype.kind in "iu":
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return a.astype(np.float32, copy=False).reshape(-1)


class ParsiGoTTS(TTSBackend):
    sample_rate = 24000

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.eng = None
        self.voice = ""
        self.lock = threading.Lock()
        self._kw_synth: dict = {}

    def load(self, plan=None, progress=None) -> None:
        root = models.ensure_tts_engine(self.cfg, progress)
        scripts = os.path.join(root, "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        cwd = os.getcwd()
        try:
            os.chdir(root)      # the reference engine resolves model/onnx relative to the repo
            import tts_onnx     # type: ignore
            cls = tts_onnx.OnnxTts
            params = inspect.signature(cls.__init__).parameters
            kw = {}
            if self.cfg.tts.seed is not None and "seed" in params:
                kw["seed"] = self.cfg.tts.seed
            for name in ("model_dir", "onnx_dir", "model_path"):
                if name in params:
                    kw[name] = os.path.join(root, "model", "onnx")
                    break
            self.eng = cls(**kw)
        finally:
            os.chdir(cwd)
        voices = os.path.join(root, "voices")
        v = self.cfg.tts.voice
        if v and os.path.exists(v):
            self.voice = v
        elif v and os.path.exists(os.path.join(voices, v if v.endswith(".wav") else v + ".wav")):
            self.voice = os.path.join(voices, v if v.endswith(".wav") else v + ".wav")
        else:
            self.voice = os.path.join(voices, "male_hello.wav")
        sp = inspect.signature(self.eng.synthesize_text).parameters
        if "mode" in sp:
            self._kw_synth["mode"] = self.cfg.tts.mode
        if "pace" in sp:
            self._kw_synth["pace"] = self.cfg.tts.pace

    def synth(self, text: str) -> np.ndarray:
        assert self.eng is not None, "TTS not loaded"
        with self.lock:
            out = self.eng.synthesize_text(text, self.voice, **self._kw_synth)
        return _as_audio(out)


class MockTTS(TTSBackend):
    """Fake voice: a soft two-tone 'hum' whose length follows the text (tests & UI demos)."""
    sample_rate = 24000

    def __init__(self, sec_per_char: float = 0.06, compute_factor: float = 0.15, sr: int = 24000):
        self.spc, self.cf, self.sample_rate = sec_per_char, compute_factor, sr
        self.calls = []

    def load(self, plan=None, progress=None) -> None: ...

    def synth(self, text: str) -> np.ndarray:
        dur = max(0.25, self.spc * len(text))
        time.sleep(dur * self.cf)
        self.calls.append(text)
        t = np.arange(int(dur * self.sample_rate)) / self.sample_rate
        env = np.minimum(1, np.minimum(t, dur - t) * 30) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.5 * t) ** 2)
        return (0.12 * env * (np.sin(2 * np.pi * 190 * t) + 0.4 * np.sin(2 * np.pi * 380 * t))).astype(np.float32)


class CachedTTS:
    """LRU phrase cache in front of any backend (fixed phrases cost nothing at runtime)."""

    def __init__(self, backend: TTSBackend, max_mb: int = 96, enabled: bool = True):
        self.backend, self.enabled = backend, enabled
        self.max_bytes = max_mb * 2**20
        self._c: "collections.OrderedDict[str, np.ndarray]" = collections.OrderedDict()
        self._bytes = 0
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self.backend.sample_rate

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

    def synth(self, text: str, cache: bool = True) -> np.ndarray:
        k = self._key(text)
        if self.enabled and cache:
            with self._lock:
                if k in self._c:
                    self._c.move_to_end(k)
                    self.hits += 1
                    return self._c[k]
        audio = self.backend.synth(text)
        self.misses += 1
        if self.enabled and cache and len(text) <= 160:
            audio16 = audio.astype(np.float16)               # halves cache RAM; inaudible loss
            with self._lock:
                self._c[k] = audio16
                self._bytes += audio16.nbytes
                while self._bytes > self.max_bytes and self._c:
                    _, v = self._c.popitem(last=False)
                    self._bytes -= v.nbytes
            return audio
        return audio

    def get(self, text: str) -> Optional[np.ndarray]:
        k = self._key(text)
        with self._lock:
            v = self._c.get(k)
        return None if v is None else v.astype(np.float32)


def create_tts(cfg: Config) -> TTSBackend:
    if cfg.tts.backend == "mock":
        return MockTTS()
    return ParsiGoTTS(cfg)
