"""Voice activity detection and turn segmentation.

    frame -> VADFrontEnd   speech probability + smoothing + adaptive noise floor
          -> Endpointer    start / pause / resume / end events for one utterance
          -> BargeIn       stricter, echo-aware detector used while the assistant talks

The endpointer is a pure state machine over 32 ms frames (no threads, no
audio devices), so it is fully unit-testable. It emits:

* ``start``  - the user began speaking
* ``pause``  - first stretch of silence (``turn.speculate_after_ms``); carries
               the audio so far, so the engine can transcribe and start
               thinking speculatively
* ``resume`` - speech came back after a pause: throw the speculation away
* ``end``    - the turn is over. The silence needed is set per pause by the
               engine from how complete the transcript looks (``set_end_silence``)
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from .audio import rms_db
from .registry import LoadContext, register

SR = 16000
FRAME = 512          # 32 ms @ 16 kHz (Silero's native window)
FRAME_MS = 32


# --------------------------------------------------------------------------- models
class VADModel:
    """Maps one 512-sample frame (16 kHz, float32) to a speech probability."""

    def __init__(self, config=None):
        self.config = config

    def load(self, ctx: LoadContext) -> None: ...
    def reset(self) -> None: ...
    def __call__(self, frame: np.ndarray) -> float:
        raise NotImplementedError


@register("vad", "silero")
class SileroVAD(VADModel):
    """Silero VAD (v4 and v5/v6 ONNX interfaces), one CPU thread, ~0.1 ms per frame."""

    def load(self, ctx: LoadContext) -> None:
        import onnxruntime as ort
        from . import models
        path = models.resolve_vad(ctx.config, ctx.progress)
        so = ort.SessionOptions()
        so.inter_op_num_threads = so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.v5 = "state" in {i.name for i in self.sess.get_inputs()}
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        if getattr(self, "v5", True):
            self.state = np.zeros((2, 1, 128), np.float32)
            self.ctx = np.zeros((1, 64), np.float32)
        else:
            self.h = np.zeros((2, 1, 64), np.float32)
            self.c = np.zeros((2, 1, 64), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32, copy=False)[None, :]
        if self.v5:
            inp = np.concatenate([self.ctx, x], axis=1)
            out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": self._sr})
            self.ctx = inp[:, -64:]
        else:
            out, self.h, self.c = self.sess.run(None, {"input": x, "sr": self._sr, "h": self.h, "c": self.c})
        return float(np.asarray(out).reshape(-1)[0])


@register("vad", "energy")
class EnergyVAD(VADModel):
    """Model-free fallback: SNR over a tracked noise floor, squashed to 0..1."""

    def load(self, ctx: LoadContext) -> None:
        self.reset()

    def reset(self) -> None:
        self.floor, self.n = -55.0, 0

    def __call__(self, frame: np.ndarray) -> float:
        db = rms_db(frame)
        self.n += 1
        a = 0.2 if self.n < 15 else 0.01
        if db < self.floor + 3 or self.n < 15:
            self.floor = (1 - a) * self.floor + a * db
        p = 1.0 / (1.0 + math.exp(-((db - self.floor) - 9.0) / 2.5))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame))))) / 2.0
        return p * (0.6 if zcr > 0.45 else 1.0)          # hiss-like noise is rarely speech


# --------------------------------------------------------------------------- front end
@dataclass
class FrameInfo:
    prob: float
    smooth: float
    db: float
    noise_db: float
    loud: bool           # above the adaptive noise gate
    on: bool             # speech by the opening threshold
    keep: bool           # speech by the (lower) holding threshold


class VADFrontEnd:
    def __init__(self, vad_cfg, model: VADModel):
        self.cfg, self.model = vad_cfg, model
        self.reset()

    def reset(self) -> None:
        self.smooth, self.noise_db, self.n = 0.0, -60.0, 0
        self.model.reset()

    def __call__(self, frame: np.ndarray) -> FrameInfo:
        c = self.cfg
        p = float(self.model(frame))
        db = rms_db(frame)
        self.n += 1
        self.smooth = c.smoothing * p + (1.0 - c.smoothing) * self.smooth
        if p < 0.2:                                        # non-speech trains the noise floor
            a = 0.15 if self.n < 30 else 0.02
            self.noise_db = (1 - a) * self.noise_db + a * db
        loud = db >= max(c.abs_min_db, self.noise_db + c.noise_gate_db)
        return FrameInfo(p, self.smooth, db, self.noise_db, loud,
                         loud and self.smooth >= c.threshold, loud and self.smooth >= c.neg_threshold)


# --------------------------------------------------------------------------- endpointer
@dataclass
class TurnEvent:
    kind: str                         # start | pause | resume | end
    utt: int                          # utterance id
    pause: int = 0                    # pause id within the utterance (pause / end)
    audio: Optional[np.ndarray] = None
    speech_ms: float = 0.0
    silence_ms: float = 0.0


class Endpointer:
    def __init__(self, vad_cfg, turn_cfg, frame_ms: int = FRAME_MS):
        self.vad, self.turn, self.fm = vad_cfg, turn_cfg, frame_ms
        self._pre_n = max(1, vad_cfg.pre_roll_ms // frame_ms)
        self._min_n = max(1, vad_cfg.min_speech_ms // frame_ms)
        self.pre: Deque[np.ndarray] = collections.deque(maxlen=self._pre_n + self._min_n + 2)
        self.utt = 0
        self._reset()

    def _reset(self) -> None:
        self.in_speech = False
        self.buf: List[np.ndarray] = []
        self.run = 0
        self.silence_ms = 0.0
        self.voiced_ms = 0.0
        self.last_voiced = 0
        self.pause_id = 0
        self.paused = False
        self.end_ms = float(self.turn.end_silence_max_ms)

    # ------------------------------------------------------------------ inspection
    @property
    def speech_ms(self) -> float:
        return len(self.buf) * self.fm

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)

    def _trimmed(self) -> np.ndarray:
        end = min(len(self.buf), self.last_voiced + 1 + self.vad.trailing_pad_ms // self.fm)
        return np.concatenate(self.buf[:end]) if end else np.zeros(0, np.float32)

    # ------------------------------------------------------------------ control
    def set_end_silence(self, utt: int, pause: int, ms: float) -> List[TurnEvent]:
        """Engine verdict for a pause (from the transcript). May end the turn right away."""
        if not (self.in_speech and self.paused and utt == self.utt and pause == self.pause_id):
            return []
        self.end_ms = float(ms)
        return [self._end()] if self.silence_ms >= self.end_ms else []

    def force_start(self, preroll: np.ndarray) -> TurnEvent:
        """Open an utterance from audio captured while the assistant was talking (barge-in)."""
        self._reset()
        self.buf = [preroll[i:i + FRAME] for i in range(0, len(preroll) - FRAME + 1, FRAME)]
        self.last_voiced = max(0, len(self.buf) - 1)
        self.voiced_ms = self.speech_ms
        self.in_speech = True
        self.utt += 1
        return TurnEvent("start", self.utt)

    def cancel(self) -> None:
        self._reset()
        self.pre.clear()

    def _end(self) -> TurnEvent:
        ev = TurnEvent("end", self.utt, self.pause_id, self._trimmed(), self.voiced_ms, self.silence_ms)
        self._reset()
        self.pre.clear()
        return ev

    # ------------------------------------------------------------------ main
    def process(self, frame: np.ndarray, info: FrameInfo) -> List[TurnEvent]:
        fm = self.fm
        if not self.in_speech:
            self.pre.append(frame)
            if info.on:
                self.run += 1
                if self.run >= self._min_n:
                    self.buf = list(self.pre)[-(self.run + self._pre_n):]
                    self.last_voiced = len(self.buf) - 1
                    self.voiced_ms = self.run * fm
                    self.in_speech = True
                    self.utt += 1
                    return [TurnEvent("start", self.utt)]
            else:
                self.run = 0
            return []

        out: List[TurnEvent] = []
        self.buf.append(frame)
        if info.keep:
            self.last_voiced = len(self.buf) - 1
            self.voiced_ms += fm
            self.silence_ms = 0.0
            if self.paused:                              # they kept talking: speculation is void
                self.paused = False
                self.end_ms = float(self.turn.end_silence_max_ms)
                out.append(TurnEvent("resume", self.utt, self.pause_id))
        else:
            self.silence_ms += fm
            if not self.paused and self.silence_ms >= self.turn.speculate_after_ms:
                self.paused = True
                self.pause_id += 1
                out.append(TurnEvent("pause", self.utt, self.pause_id, self._trimmed(),
                                     self.voiced_ms, self.silence_ms))
            if self.silence_ms >= self.end_ms:
                out.append(self._end())
                return out
        if self.speech_ms >= self.turn.max_utterance_s * 1000:
            out.append(self._end())
        return out


# --------------------------------------------------------------------------- barge-in
class BargeInDetector:
    """Is the sound picked up *while the assistant speaks* the user talking over it?

    Without acoustic echo cancellation the microphone hears our own voice. We
    know what we play, so we learn the speaker->mic coupling (mic dB minus
    playback dB) from frames that did not trigger, and require the mic to beat
    that expected echo by ``echo_margin_db``.
    """

    def __init__(self, turn_cfg, frame_ms: int = FRAME_MS):
        self.cfg, self.fm = turn_cfg, frame_ms
        self.coupling: Optional[float] = None
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, info: FrameInfo, playback_db: Optional[float]) -> bool:
        c = self.cfg
        speechy = info.loud and info.smooth >= c.barge_in_threshold
        if playback_db is not None and playback_db > -50.0:
            ratio = info.db - playback_db
            if self.coupling is None:
                self.coupling = ratio
            hit = speechy and ratio >= self.coupling + c.echo_margin_db
            if not hit:                                   # track the echo path slowly
                self.coupling = 0.95 * self.coupling + 0.05 * min(ratio, self.coupling + 3.0)
        else:
            hit = speechy
        self.count = self.count + 1 if hit else 0
        return self.count * self.fm >= c.barge_in_min_ms
