"""Professional-grade VAD stack.

    frame -> VADFrontEnd (Silero probability + EMA + adaptive noise floor + gate)
          -> Endpointer   (hysteresis, pre-roll, min-speech, two-stage end-of-turn)
          -> BargeInDetector (stricter, echo-aware, used while the agent talks)

The Endpointer is a pure state machine over frames, so it is unit-testable
without audio hardware.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from .config import BargeInConfig, VADConfig

SR = 16000
FRAME = 512  # 32 ms @ 16 kHz


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float32)) + 1e-12))
    return 20.0 * math.log10(r + 1e-9)


# ------------------------------------------------------------------ models
class SileroONNX:
    """Silero VAD (v4 and v5/v6 ONNX interfaces) on onnxruntime, single CPU thread."""

    def __init__(self, path: str):
        import onnxruntime as ort  # lazy
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        self.v5 = "state" in names
        self.ctx = 64
        self.reset()

    def reset(self) -> None:
        if self.v5:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
            self.context = np.zeros((1, self.ctx), dtype=np.float32)
        else:
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32, copy=False)[None, :]
        sr = np.array(SR, dtype=np.int64)
        if self.v5:
            inp = np.concatenate([self.context, x], axis=1)
            out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": sr})
            self.context = inp[:, -self.ctx:]
        else:
            out, self.h, self.c = self.sess.run(None, {"input": x, "sr": sr, "h": self.h, "c": self.c})
        return float(np.asarray(out).reshape(-1)[0])


class EnergyModel:
    """Fallback pseudo-probability from SNR over a tracked noise floor (no model file needed)."""

    def __init__(self):
        self.floor = -55.0
        self.n = 0

    def reset(self) -> None:
        self.floor, self.n = -55.0, 0

    def __call__(self, frame: np.ndarray) -> float:
        db = rms_db(frame)
        self.n += 1
        a = 0.2 if self.n < 15 else 0.01
        if db < self.floor + 3 or self.n < 15:
            self.floor = (1 - a) * self.floor + a * db
        snr = db - self.floor
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame))))) / 2.0
        p = 1.0 / (1.0 + math.exp(-(snr - 9.0) / 2.5))
        if zcr > 0.45:      # hiss / fricative-only noise
            p *= 0.6
        return p


def load_vad_model(cfg: VADConfig, model_path: Optional[str] = None):
    if cfg.backend == "energy" or not model_path:
        return EnergyModel()
    return SileroONNX(model_path)


# ------------------------------------------------------------------ front end
@dataclass
class FrameInfo:
    prob: float
    smooth: float
    rms_db: float
    noise_db: float
    gated: bool          # loud enough above the noise floor
    on: bool             # speech-like using the ON threshold
    off: bool            # speech-like using the (lower) OFF threshold


class VADFrontEnd:
    def __init__(self, cfg: VADConfig, model):
        self.cfg, self.model = cfg, model
        self.reset()

    def reset(self) -> None:
        self.smooth = 0.0
        self.noise_db = -60.0
        self.n = 0
        if hasattr(self.model, "reset"):
            self.model.reset()

    def __call__(self, frame: np.ndarray) -> FrameInfo:
        c = self.cfg
        p = float(self.model(frame))
        db = rms_db(frame)
        self.n += 1
        self.smooth = c.smoothing * p + (1.0 - c.smoothing) * self.smooth
        if p < 0.2:    # quiet/non-speech frames train the noise floor (fast at start, slow later)
            a = 0.15 if self.n < 30 else 0.02
            self.noise_db = (1 - a) * self.noise_db + a * db
        gated = db >= max(c.abs_min_db, self.noise_db + c.noise_gate_db)
        return FrameInfo(p, self.smooth, db, self.noise_db, gated,
                         gated and self.smooth >= c.threshold_on,
                         gated and self.smooth >= c.threshold_off)


# ------------------------------------------------------------------ endpointer
@dataclass
class TurnEvent:
    kind: str                       # "start" | "possible_end" | "end"
    audio: Optional[np.ndarray] = None
    seq: int = 0
    speech_ms: float = 0.0
    silence_ms: float = 0.0         # trailing silence already elapsed when the event fired


class Endpointer:
    def __init__(self, cfg: VADConfig, frame_ms: int = 32):
        self.cfg, self.frame_ms = cfg, frame_ms
        self._pre_frames = max(1, cfg.pre_roll_ms // frame_ms)
        self._min_frames = max(1, cfg.min_speech_ms // frame_ms)
        self.pre: Deque[np.ndarray] = collections.deque(maxlen=self._pre_frames + self._min_frames + 2)
        self.seq = 0
        self.reset()

    # -- state ----------------------------------------------------------
    def reset(self) -> None:
        self.in_speech = False
        self.buf: List[np.ndarray] = []
        self.run = 0
        self.silence_ms = 0.0
        self.last_speech_idx = 0
        self.possible_sent = False
        self.pre.clear()

    @property
    def speech_ms(self) -> float:
        return len(self.buf) * self.frame_ms

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)

    def _trimmed(self) -> np.ndarray:
        pad = self.cfg.trailing_pad_ms // self.frame_ms
        end = min(len(self.buf), self.last_speech_idx + 1 + pad)
        return np.concatenate(self.buf[:end]) if end else np.zeros(0, np.float32)

    # -- control --------------------------------------------------------
    def force_start(self, preroll: np.ndarray) -> TurnEvent:
        """Begin a turn from audio captured while the agent was talking (barge-in)."""
        self.reset()
        frames = [preroll[i:i + FRAME] for i in range(0, len(preroll) - FRAME + 1, FRAME)]
        self.buf = [f.astype(np.float32, copy=False) for f in frames] or []
        self.last_speech_idx = max(0, len(self.buf) - 1)
        self.in_speech = True
        self.seq += 1
        return TurnEvent("start", seq=self.seq)

    def finalize(self, seq: int) -> List[TurnEvent]:
        """Called after an early decode decided the utterance is complete."""
        if self.in_speech and self.possible_sent and seq == self.seq:
            return [self._end()]
        return []

    def _end(self) -> TurnEvent:
        ev = TurnEvent("end", self._trimmed(), self.seq, self.speech_ms, self.silence_ms)
        self.reset()
        self.seq += 1
        return ev

    # -- main -----------------------------------------------------------
    def process(self, frame: np.ndarray, info: FrameInfo) -> List[TurnEvent]:
        c, fm = self.cfg, self.frame_ms
        out: List[TurnEvent] = []
        self.pre.append(frame)
        if not self.in_speech:
            if info.on:
                self.run += 1
                if self.run >= self._min_frames:
                    take = self.run + self._pre_frames
                    self.buf = list(self.pre)[-take:]
                    self.last_speech_idx = len(self.buf) - 1
                    self.in_speech = True
                    self.silence_ms = 0.0
                    self.possible_sent = False
                    self.seq += 1
                    out.append(TurnEvent("start", seq=self.seq))
            else:
                self.run = 0
            return out

        self.buf.append(frame)
        if info.off:
            self.last_speech_idx = len(self.buf) - 1
            self.silence_ms = 0.0
            if self.possible_sent:          # user resumed: invalidate any pending early decode
                self.possible_sent = False
                self.seq += 1
        else:
            self.silence_ms += fm
            if (c.adaptive_endpointing and not self.possible_sent and self.silence_ms >= c.fast_silence_ms
                    and self.silence_ms < c.silence_ms):
                self.possible_sent = True
                out.append(TurnEvent("possible_end", self._trimmed(), self.seq, self.speech_ms, self.silence_ms))
            if self.silence_ms >= c.silence_ms:
                out.append(self._end())
                return out
        if self.speech_ms >= c.max_utterance_s * 1000:
            out.append(self._end())
        return out


# ------------------------------------------------------------------ barge-in
class BargeInDetector:
    """Decides whether sound picked up *while the agent speaks* is the user talking over it.

    Echo handling: we know what we play.  We learn the speaker->mic coupling
    (mic_dB - playback_dB) from frames that did not trigger, and require the mic
    to exceed that expected echo by ``echo_margin_db``.
    """

    def __init__(self, cfg: BargeInConfig, vad_cfg: VADConfig, frame_ms: int = 32):
        self.cfg, self.vad, self.frame_ms = cfg, vad_cfg, frame_ms
        self.ratio: Optional[float] = None
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, info: FrameInfo, playback_db: Optional[float]) -> bool:
        c = self.cfg
        if not c.enabled:
            return False
        base = info.smooth >= c.threshold and info.gated
        if playback_db is not None and playback_db > -55.0:
            ratio = info.rms_db - playback_db
            if self.ratio is None:
                self.ratio = ratio
            cond = base and ratio >= self.ratio + c.echo_margin_db
            if not cond:
                self.ratio = 0.96 * self.ratio + 0.04 * min(ratio, self.ratio + 3.0)
        else:
            cond = base
        self.count = self.count + 1 if cond else 0
        return self.count * self.frame_ms >= c.min_speech_ms
