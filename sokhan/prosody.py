"""Voice-quality / prosody sensing (pure numpy, ~5 ms per second of audio on CPU).

Extracts pitch (F0), energy, pace, pauses, voice-quality proxies and end-of-
utterance intonation, compares them with a running per-speaker baseline, and
turns the result into (a) numbers and (b) a compact cue string for the LLM.

This is *heuristic paralinguistics*, not emotion recognition.  For a learned
model plug an ``EmotionModel`` (see ``EmotionModel`` protocol) into the analyzer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Protocol

import numpy as np

from .config import ProsodyConfig

SR = 16000


class EmotionModel(Protocol):
    """Optional plug-in: e.g. a wav2vec2/emotion2vec ONNX head trained on ShEMO."""
    def predict(self, audio: np.ndarray, sr: int) -> Dict[str, float]: ...


@dataclass
class ProsodyFeatures:
    duration_s: float = 0.0
    voiced_ratio: float = 0.0
    f0_mean: float = 0.0
    f0_std_st: float = 0.0          # semitones
    f0_range_st: float = 0.0        # p90-p10, semitones
    f0_slope_st_s: float = 0.0      # global trend, semitones / s
    end_slope_st: float = 0.0       # last ~300 ms vs before (semitones) -> question intonation
    rms_db_mean: float = -120.0
    rms_db_peak: float = -120.0
    rms_db_std: float = 0.0
    syllable_rate: float = 0.0      # nuclei / s of active speech
    words_per_s: float = 0.0
    pause_ratio: float = 0.0
    n_pauses: int = 0
    longest_pause_s: float = 0.0
    hnr_db: float = 0.0
    jitter_pct: float = 0.0
    shimmer_pct: float = 0.0
    centroid_hz: float = 0.0
    # interpreted
    arousal: float = 0.5            # 0 calm .. 1 activated
    energy: str = "normal"          # low | normal | high
    pace: str = "normal"            # slow | normal | fast
    pitch_var: str = "normal"       # flat | normal | animated
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Baseline:
    n: int = 0
    f0: float = 0.0
    f0_std: float = 0.0
    rms: float = 0.0
    rate: float = 0.0
    pauses: float = 0.0


def _hz_to_st(f: np.ndarray, ref: float = 100.0) -> np.ndarray:
    return 12.0 * np.log2(np.maximum(f, 1e-3) / ref)


def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop]


class ProsodyAnalyzer:
    def __init__(self, cfg: Optional[ProsodyConfig] = None, emotion_model: Optional[EmotionModel] = None):
        self.cfg = cfg or ProsodyConfig()
        self.model = emotion_model
        self.base = _Baseline()

    def reset_baseline(self) -> None:
        self.base = _Baseline()

    # ------------------------------------------------------------ analysis
    def analyze(self, audio: np.ndarray, sr: int = SR, text: Optional[str] = None,
                update_baseline: bool = True) -> ProsodyFeatures:
        x = np.asarray(audio, dtype=np.float32)
        ft = ProsodyFeatures(duration_s=len(x) / sr)
        if len(x) < int(0.25 * sr):
            return ft
        x = x - float(np.mean(x))
        n, hop = int(0.040 * sr), int(0.010 * sr)
        fr = _frames(x, n, hop)
        win = np.hanning(n).astype(np.float32)
        rms = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-9)
        active_thr = max(db.max() - 35.0, -60.0)
        active = db > active_thr
        if active.sum() < 5:
            return ft

        # --- pitch via FFT autocorrelation (all frames at once)
        nfft = 2048
        spec = np.fft.rfft(fr * win, n=nfft, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, : n // 2 + 1]
        ac0 = ac[:, :1] + 1e-9
        acn = ac / ac0
        lo, hi = int(sr / 400), int(sr / 70)
        seg = acn[:, lo:hi]
        pk = np.argmax(seg, axis=1)
        strength = seg[np.arange(len(seg)), pk]
        lag = (pk + lo).astype(np.float64)
        # parabolic refinement
        idx = np.clip(pk + lo, 1, acn.shape[1] - 2)
        a, b, c = acn[np.arange(len(acn)), idx - 1], acn[np.arange(len(acn)), idx], acn[np.arange(len(acn)), idx + 1]
        denom = (a - 2 * b + c)
        shift = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
        lag = lag + np.clip(shift, -1, 1)
        f0 = sr / lag
        voiced = active & (strength > 0.45)
        # kill octave/outlier jumps with a 5-frame median
        f0v = np.where(voiced, f0, np.nan)
        f0s = _nanmedian_filter(f0v, 5)
        voiced = voiced & ~np.isnan(f0s)
        ft.voiced_ratio = float(voiced.sum() / max(1, active.sum()))

        if voiced.sum() >= 6:
            vf0 = f0s[voiced]
            st = _hz_to_st(vf0)
            ft.f0_mean = float(np.mean(vf0))
            ft.f0_std_st = float(np.std(st))
            ft.f0_range_st = float(np.percentile(st, 90) - np.percentile(st, 10))
            t = np.nonzero(voiced)[0] * (hop / sr)
            if t[-1] - t[0] > 0.3:
                ft.f0_slope_st_s = float(np.polyfit(t, st, 1)[0])
            k = max(5, int(0.3 * sr / hop))
            v_idx = np.nonzero(voiced)[0]
            if len(v_idx) > 2 * k:
                tail = _hz_to_st(f0s[v_idx[-k:]])
                prev = _hz_to_st(f0s[v_idx[-3 * k:-k]])
                ft.end_slope_st = float(np.median(tail) - np.median(prev))
            # voice-quality proxies
            per = 1.0 / vf0
            if len(per) > 3:
                ft.jitter_pct = float(np.mean(np.abs(np.diff(per))) / np.mean(per) * 100)
            amp = rms[voiced]
            if len(amp) > 3:
                ft.shimmer_pct = float(np.mean(np.abs(np.diff(amp))) / np.mean(amp) * 100)
            r = np.clip(strength[voiced], 1e-3, 0.999)
            ft.hnr_db = float(np.clip(np.mean(10 * np.log10(r / (1 - r))), 0, 30))

        # --- energy
        adb = db[active]
        ft.rms_db_mean = float(np.mean(adb))
        ft.rms_db_peak = float(np.percentile(adb, 95))
        ft.rms_db_std = float(np.std(adb))

        # --- spectral centroid over active frames
        mag = np.abs(spec[active])
        freqs = np.fft.rfftfreq(nfft, 1 / sr)
        ft.centroid_hz = float(np.mean((mag * freqs).sum(1) / (mag.sum(1) + 1e-9)))

        # --- pauses (internal silences >= 220 ms between first and last active frame)
        idx_act = np.nonzero(active)[0]
        first, last = idx_act[0], idx_act[-1]
        inner = active[first:last + 1]
        runs, cur = [], 0
        for v in inner:
            if not v:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        min_run = int(0.22 * sr / hop)
        pauses = [r * hop / sr for r in runs if r >= min_run]
        span = max(1e-3, (last - first + 1) * hop / sr)
        ft.n_pauses = len(pauses)
        ft.longest_pause_s = float(max(pauses)) if pauses else 0.0
        ft.pause_ratio = float(sum(pauses) / span)

        # --- syllable-rate proxy: energy-envelope peaks with >=3 dB prominence
        env = np.convolve(db, np.ones(5) / 5, mode="same")
        peaks = _count_peaks(env, prominence=3.0, floor=active_thr + 3.0, min_gap=int(0.09 * sr / hop))
        speaking_time = max(0.3, span - sum(pauses))
        ft.syllable_rate = float(peaks / speaking_time)
        if text:
            words = len(text.split())
            ft.words_per_s = float(words / max(0.3, span))

        self._interpret(ft)
        if self.model is not None:
            try:
                ft.extra.update({f"emo_{k}": float(v) for k, v in self.model.predict(x, sr).items()})
            except Exception:
                pass
        if update_baseline and ft.confidence >= 0.3:
            self._update_baseline(ft)
        return ft

    # ------------------------------------------------------------ baseline / interpretation
    def _update_baseline(self, ft: ProsodyFeatures) -> None:
        b, a = self.base, self.cfg.baseline_alpha
        if b.n == 0:
            b.f0, b.f0_std, b.rms, b.rate, b.pauses = ft.f0_mean, ft.f0_std_st, ft.rms_db_mean, ft.syllable_rate, ft.pause_ratio
        else:
            b.f0 = (1 - a) * b.f0 + a * (ft.f0_mean or b.f0)
            b.f0_std = (1 - a) * b.f0_std + a * ft.f0_std_st
            b.rms = (1 - a) * b.rms + a * ft.rms_db_mean
            b.rate = (1 - a) * b.rate + a * (ft.syllable_rate or b.rate)
            b.pauses = (1 - a) * b.pauses + a * ft.pause_ratio
        b.n += 1

    def _interpret(self, ft: ProsodyFeatures) -> None:
        b = self.base
        have_base = b.n >= 2
        dur_ok = min(1.0, ft.duration_s / 2.0)
        voiced_ok = min(1.0, ft.voiced_ratio / 0.5)
        ft.confidence = float(0.25 + 0.75 * dur_ok * voiced_ok) * (1.0 if have_base else 0.75)

        # z-like deviations (baseline if available, else broad population priors)
        d_rms = ft.rms_db_mean - (b.rms if have_base else -28.0)
        d_rate = ft.syllable_rate - (b.rate if have_base else 4.6)
        d_pitch = (12 * math.log2(ft.f0_mean / b.f0)) if (have_base and ft.f0_mean and b.f0) else 0.0
        d_var = ft.f0_std_st - (b.f0_std if have_base else 2.5)

        ft.energy = "high" if d_rms > 4.5 else "low" if d_rms < -5.5 else "normal"
        ft.pace = "fast" if d_rate > 1.1 else "slow" if d_rate < -1.1 else "normal"
        ft.pitch_var = "animated" if d_var > 1.4 else "flat" if d_var < -1.2 else "normal"

        z = 0.10 * d_rms + 0.28 * d_rate + 0.16 * d_pitch + 0.18 * d_var
        ft.arousal = float(1.0 / (1.0 + math.exp(-z)))

        tags: List[str] = []
        if ft.energy == "high":
            tags.append("loud" if d_rms > 8 else "high energy")
        elif ft.energy == "low":
            tags.append("quiet" if d_rms < -9 else "low energy")
        if ft.pace != "normal":
            tags.append(f"{ft.pace} pace")
        if ft.pitch_var == "animated":
            tags.append("expressive pitch")
        elif ft.pitch_var == "flat":
            tags.append("flat pitch")
        if ft.end_slope_st > 1.6 and ft.duration_s > 0.6:
            tags.append("rising intonation")
        elif ft.end_slope_st < -2.2 and ft.duration_s > 0.9:
            tags.append("falling intonation")
        if ft.n_pauses >= 2 or ft.longest_pause_s > 0.7 or (have_base and ft.pause_ratio > b.pauses + 0.18):
            tags.append("hesitant")
        if ft.jitter_pct > 4.0 and ft.hnr_db < 8 and ft.voiced_ratio > 0.4:
            tags.append("strained voice")
        if ft.arousal > 0.72 and ft.energy != "low":
            tags.append("agitated or excited")
        elif ft.arousal < 0.28 and ft.energy != "high":
            tags.append("subdued")
        ft.tags = tags

    # ------------------------------------------------------------ output
    def describe(self, ft: ProsodyFeatures) -> str:
        """Compact cue string for the prompt; empty when there is nothing notable."""
        if ft.confidence < self.cfg.min_confidence or not ft.tags:
            return ""
        return "[voice cues: " + ", ".join(ft.tags[:4]) + "]"


# ---------------------------------------------------------------- helpers
def _nanmedian_filter(v: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(v, np.nan)
    h = k // 2
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        w = v[max(0, i - h): i + h + 1]
        w = w[~np.isnan(w)]
        m = np.median(w)
        # reject octave jumps against the local median
        if abs(12 * math.log2(v[i] / m)) < 5.0:
            out[i] = v[i]
    return out


def _count_peaks(env: np.ndarray, prominence: float, floor: float, min_gap: int) -> int:
    n, count, last = len(env), 0, -10 ** 9
    i = 1
    while i < n - 1:
        if env[i] > floor and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            lo_l = env[max(0, i - 12): i].min() if i > 0 else env[i]
            lo_r = env[i + 1: i + 13].min() if i + 1 < n else env[i]
            if env[i] - max(lo_l, lo_r) >= prominence and i - last >= min_gap:
                count += 1
                last = i
        i += 1
    return count
