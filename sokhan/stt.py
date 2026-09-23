"""Speech-to-text backends.

``sherpa_onnx`` (default) runs any sherpa-onnx offline recognizer on CPU:
NeMo CTC (default: Shenava, Persian), Whisper (multilingual), Zipformer /
Transducer, Paraformer and SenseVoice - pick one with ``stt.model`` +
``stt.model_type``. Utterances are short and these models run 30-150x faster
than realtime, so each VAD segment is decoded in one shot (plus cheap live
partials while the user is still talking).
"""
from __future__ import annotations

import glob
import importlib.util
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .audio import resample, to_float32
from .lang import get_language
from .registry import LoadContext, register

log = logging.getLogger("sokhan.stt")
SR = 16000


@dataclass
class STTResult:
    text: str
    duration_s: float = 0.0
    decode_s: float = 0.0


class STTBackend:
    """Implement ``transcribe``; everything else is optional."""
    sample_rate = SR

    def __init__(self, config=None):
        self.config = config

    def load(self, ctx: LoadContext) -> None: ...

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> STTResult:
        raise NotImplementedError

    def warmup(self) -> None:
        self.transcribe(np.zeros(SR // 2, np.float32))

    def close(self) -> None: ...


@register("stt", "sherpa_onnx", "sherpa", "sherpa_nemo_ctc")
class SherpaOnnxSTT(STTBackend):
    def __init__(self, config):
        super().__init__(config)
        self.rec = None
        self._lock = threading.Lock()
        self._post: Optional[Callable[[str], str]] = None

    # ------------------------------------------------------------------ load
    def load(self, ctx: LoadContext) -> None:
        try:
            import sherpa_onnx  # type: ignore
        except ImportError as e:
            raise RuntimeError("the default STT needs `pip install sherpa-onnx`") from e
        from . import models
        c, cfg = self.config.stt, self.config
        d = models.resolve_stt(cfg, ctx.progress)
        threads = c.threads or ctx.plan.stt_threads
        provider = "cuda" if "CUDAExecutionProvider" in ctx.plan.ort_providers else "cpu"
        lang = c.language or cfg.language
        kind = c.model_type.lower()

        def f(*patterns: str) -> str:
            for p in patterns:
                hits = sorted(glob.glob(os.path.join(d, p)), key=lambda x: (".int8." in x, len(x)))
                hits = [h for h in hits if ".q4." not in h]
                if hits:
                    return self._quant(hits, ctx)
            raise FileNotFoundError(f"{c.model}: no file matching {patterns} in {d}")

        tokens = f("tokens.txt", "*tokens.txt")
        common = dict(num_threads=threads, provider=provider)
        if kind == "nemo_ctc":
            build = lambda **k: sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=f("model.onnx", "*.onnx"),
                                                                             tokens=tokens, **k)
        elif kind == "zipformer_ctc":
            build = lambda **k: sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(model=f("model.onnx", "*.onnx"),
                                                                                  tokens=tokens, **k)
        elif kind == "whisper":
            build = lambda **k: sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=f("*encoder*.onnx"), decoder=f("*decoder*.onnx"), tokens=tokens,
                language=lang, task="transcribe", **k)
        elif kind == "transducer":
            build = lambda **k: sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=f("*encoder*.onnx"), decoder=f("*decoder*.onnx"), joiner=f("*joiner*.onnx"),
                tokens=tokens, **k)
        elif kind == "paraformer":
            build = lambda **k: sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=f("model.onnx", "*.onnx"), tokens=tokens, **k)
        elif kind == "sense_voice":
            build = lambda **k: sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=f("model.onnx", "*.onnx"), tokens=tokens, language=lang if lang in
                ("zh", "en", "ja", "ko", "yue") else "auto", use_itn=True, **k)
        else:
            raise ValueError(f"unknown stt.model_type {c.model_type!r}")
        try:
            self.rec = build(**common)
        except Exception:
            if provider == "cpu":
                raise
            log.warning("STT on %s failed; falling back to CPU", provider)
            self.rec = build(num_threads=threads, provider="cpu")

        # transcript post-processing: the model's own ITN if it ships one, else the language pack's
        pack = get_language(lang)
        self._post = pack.itn
        own = os.path.join(d, "persian_itn.py")
        if c.itn and os.path.exists(own) and lang == "fa":
            try:
                spec = importlib.util.spec_from_file_location("sokhan_model_itn", own)
                mod = importlib.util.module_from_spec(spec)            # type: ignore[arg-type]
                spec.loader.exec_module(mod)                            # type: ignore[union-attr]
                model_itn = mod.itn
                self._post = lambda t, _m=model_itn, _p=pack.pre_normalize: _m(_p(t) if _p else t)
            except Exception as e:
                log.debug("model ITN unavailable (%s)", e)
        if not c.itn:
            self._post = pack.pre_normalize

    def _quant(self, hits: List[str], ctx: LoadContext) -> str:
        from . import models
        q = self.config.stt.quant.lower()
        int8 = [h for h in hits if ".int8." in h]
        plain = [h for h in hits if ".int8." not in h]
        if q in ("int8", "q8") and int8:
            return int8[0]
        if q in ("q4", "int4") and plain and plain[0].endswith(".onnx"):
            return models.pick_quant(plain[0], "q4", ctx.progress, "stt")
        return plain[0] if plain else hits[0]

    # ------------------------------------------------------------------ run
    def _prep(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = resample(to_float32(audio), sr, SR)
        c = self.config.stt
        if c.auto_gain and len(x):
            db = 20 * np.log10(float(np.sqrt(np.mean(x * x))) + 1e-9)
            if db > -50:
                x = np.clip(x * float(np.clip(10 ** ((c.target_dbfs - db) / 20), 0.5, 10.0)), -1, 1)
        pad = np.zeros(int(0.2 * SR), np.float32)       # context for the first/last phoneme
        return np.concatenate([pad, x, pad]).astype(np.float32)

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> STTResult:
        assert self.rec is not None, "STT not loaded"
        dur = len(audio) / sr
        if dur * 1000 < self.config.stt.min_audio_ms:
            return STTResult("", dur)
        x = self._prep(audio, sr)
        t = time.perf_counter()
        with self._lock:
            s = self.rec.create_stream()
            s.accept_waveform(SR, x)
            self.rec.decode_stream(s)
            text = s.result.text.strip()
        if text and self._post:
            try:
                text = self._post(text)
            except Exception:
                pass
        return STTResult(text.strip(), dur, time.perf_counter() - t)


@register("stt", "mock")
class MockSTT(STTBackend):
    """Deterministic STT for tests and demos: ``script[i]`` or ``fn(audio)``."""

    def __init__(self, config=None, script=None, fn: Optional[Callable[[np.ndarray], str]] = None,
                 delay: float = 0.01):
        super().__init__(config)
        self.script, self.fn, self.delay, self.i, self.calls = list(script or []), fn, delay, 0, 0

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> STTResult:
        self.calls += 1
        time.sleep(self.delay)
        if self.fn:
            text = self.fn(audio)
        elif self.script:
            text = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
        else:
            text = "hello"
        return STTResult(text, len(audio) / sr, self.delay)
