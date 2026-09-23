"""Text-to-speech backends.

``pocket_tts`` (default)  Pocket-TTS Farsi v2, pure ONNX, CPU. Streams 80 ms audio
                          blocks while it generates (first sound in ~0.1 s) and
                          clones a voice from a ~5 s sample.
``sherpa_onnx``           any sherpa-onnx TTS model: Piper/VITS voices for dozens of
                          languages, Kokoro, Matcha. Very light, no cloning.
``mock``                  a soft hum whose length follows the text (tests, UI work).

A backend turns text into a *plan* (``prepare``: text front-end, e.g. G2P)
and a plan into audio blocks (``stream``). The engine runs the two stages in
separate threads so the front-end of the next sentence overlaps with the
audio of the current one.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

from .audio import read_audio, resample, to_float32
from .config import cache_root
from .registry import LoadContext, register

log = logging.getLogger("sokhan.tts")
VoiceRef = Union[str, np.ndarray]


class VoiceCloningNotSupported(RuntimeError):
    pass


class TTSBackend:
    """Implement ``stream`` (or just ``synthesize``); the rest is optional."""
    sample_rate: int = 24000
    supports_cloning: bool = False
    supports_streaming: bool = False

    def __init__(self, config=None):
        self.config = config

    def load(self, ctx: LoadContext) -> None: ...

    def prepare(self, text: str) -> Any:
        """Text front-end (normalisation, G2P). Runs ahead of ``stream``."""
        return text

    def stream(self, plan: Any, cancel: threading.Event) -> Iterator[np.ndarray]:
        yield self.synthesize(plan)

    def synthesize(self, text: str) -> np.ndarray:
        blocks = list(self.stream(self.prepare(text), threading.Event()))
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def set_voice(self, ref: VoiceRef, sr: Optional[int] = None) -> None:
        raise VoiceCloningNotSupported(f"{type(self).__name__} cannot clone voices")

    def voices(self) -> List[str]:
        return []

    def warmup(self) -> None:
        self.synthesize("Hello.")

    def close(self) -> None: ...


# =========================================================================== Pocket-TTS
@register("tts", "pocket_tts", "pocket", "parsigo_onnx")
class PocketTTS(TTSBackend):
    sample_rate = 24000
    supports_cloning = True
    supports_streaming = True
    BUILTIN_VOICES = ("female_narration", "male_news", "male_hello")
    QUIET = 0.012                 # block RMS below this is silence
    SPOKEN = 0.02                 # block RMS above this is speech

    def __init__(self, config):
        super().__init__(config)
        self.voice: Optional[Tuple[np.ndarray, int]] = None
        self.voice_name = ""
        self._level = config.tts.loudness or 0.1
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sokhan-tts-dec")

    # ------------------------------------------------------------------ load
    def load(self, ctx: LoadContext) -> None:
        import onnxruntime as ort
        import sentencepiece as spm
        from . import models
        cfg = self.config
        root = self.root = models.resolve_pocket_tts(cfg, ctx.progress)
        scripts = os.path.join(root, "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import tts_onnx  # type: ignore  (text front-end helpers of the reference engine)
        from g2p_onnx import OnnxG2P  # type: ignore
        self._ref = tts_onnx
        onnx_dir = os.path.join(root, "model", "onnx")
        with open(os.path.join(onnx_dir, "manifest.json"), encoding="utf-8") as f:
            c = json.load(f)["constants"]
        self.ldim, self.dim, self.L = c["ldim"], c["dim"], c["layers"]
        self.H, self.D, self.CAP = c["heads"], c["dim_per_head"], c["cache_capacity"]
        self.sample_rate, self.frame_rate = c["sample_rate"], c["frame_rate"]
        self.spl = c["mimi_steps_per_latent"]
        self.tps, self.pad_s = c["tokens_per_second_estimate"], c["gen_seconds_padding"]
        w = np.load(os.path.join(onnx_dir, "weights.npz"))
        self.lut, self.spk_proj = w["lut_weight"], w["speaker_proj"]
        self.bos_voice = w["bos_before_voice"][0]
        self.emb_std, self.emb_mean = w["emb_std"], w["emb_mean"]
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(root, "model", "v2", "tokenizer_ph.model"))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, ctx.plan.tts_threads)
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self._so, self._prov = so, ctx.plan.ort_providers
        flow = models.pick_quant(os.path.join(onnx_dir, "flow_lm_step.onnx"), cfg.tts.quant, ctx.progress, "tts")
        self.quant = "q4" if flow.endswith(".q4.onnx") else "fp32"
        ctx.report("tts", 0.9, "loading voice model")
        self.s_flow = ort.InferenceSession(flow, so, providers=self._prov)
        self.s_dec = ort.InferenceSession(os.path.join(onnx_dir, "mimi_decoder_step_kv.onnx"), so,
                                          providers=self._prov)
        self.s_enc = None                                  # loaded on demand (new voices only)
        self.dec_in = [i.name for i in self.s_dec.get_inputs()]
        self.dec_out = [o.name for o in self.s_dec.get_outputs()]
        init = np.load(os.path.join(onnx_dir, "decode_state_init.npz"))
        self.dec_init = [init[k] for k in init.files]
        n_state = len(self.dec_in) - 1
        self.cache_slots = [i for i, k in enumerate(init.files) if k.endswith(".cache")]
        self.small_slots = [i for i in range(n_state) if i not in self.cache_slots]
        self.out_small = [i for i, n in enumerate(self.dec_out) if n.endswith("o") and n != "audio"]
        self.out_kv = [i for i, n in enumerate(self.dec_out) if n.startswith("kv")]

        from pathlib import Path
        self.g2p = OnnxG2P(Path(onnx_dir))
        self._chunker = object.__new__(tts_onnx.OnnxTts)   # only its phoneme chunking helpers are used
        self._chunker.sp = self.sp
        self.rng = np.random.default_rng(cfg.tts.seed)
        self.set_voice(cfg.tts.voice or self.BUILTIN_VOICES[0])

    def voices(self) -> List[str]:
        return list(self.BUILTIN_VOICES)

    # ------------------------------------------------------------------ voices
    def set_voice(self, ref: VoiceRef, sr: Optional[int] = None) -> None:
        """Clone a voice from ~3-5 s of clean speech (path or array). Cached on disk."""
        if isinstance(ref, str):
            name = ref
            path = ref
            if ref in self.BUILTIN_VOICES or not os.path.exists(ref):
                path = os.path.join(self.root, "voices", ref if ref.endswith(".wav") else ref + ".wav")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"voice {ref!r}: not a file and not one of {self.BUILTIN_VOICES}")
            audio, sr = read_audio(path)
        else:
            if sr is None:
                raise ValueError("pass sr= with a numpy voice sample")
            audio, name = to_float32(ref), "custom"
        audio = resample(audio, int(sr), self.sample_rate)[: 5 * self.sample_rate]
        audio, _ = self._ref.trim_hot_onset(audio, self.sample_rate)
        if len(audio) < self.sample_rate:
            raise ValueError("voice sample too short: record at least 1-2 s (3-5 s is ideal)")
        key = hashlib.sha1(audio.astype(np.float32).tobytes() + self.quant.encode()).hexdigest()[:16]
        cpath = os.path.join(cache_root(self.config), "tts", "voices", key + ".npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            off = int(z["off"])
            cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
            cache[:, :, :off] = z["kv"]
        else:
            import onnxruntime as ort
            if self.s_enc is None:
                enc = os.path.join(self.root, "model", "onnx", "mimi_encoder.onnx")
                self.s_enc = ort.InferenceSession(enc, self._so, providers=self._prov)
            lat = self.s_enc.run(None, {"audio": audio[None, None, :].astype(np.float32)})[0]
            cond = (lat[0] @ self.spk_proj.T).astype(np.float32)
            emb = np.concatenate([self.bos_voice, cond])[None]
            cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
            _, _, off = self._flow(np.zeros((1, 0, self.ldim), np.float32), emb, 0,
                                   np.zeros((1, self.ldim), np.float32), cache)
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            np.savez(cpath, kv=cache[:, :, :off], off=off)
        with self._lock:
            self.voice, self.voice_name = (cache, off), name

    # ------------------------------------------------------------------ front-end
    def prepare(self, text: str) -> List[Tuple[str, float]]:
        """Text -> [(phoneme chunk, pause before it in s)] using the reference text planner."""
        t, ref = self.config.tts, self._ref
        jobs: List[Tuple[str, float]] = []
        for sent in ref.split_sentences(text):
            plan = ref.pack_phrases(ref.plan_phrases(sent, self.g2p, self.sp))
            first_in_sentence = True
            for ph, _gap in plan:
                for ci, chunk in enumerate(self._chunker.chunk_phonemes(ph)):
                    if not jobs:
                        gap = 0.0
                    elif first_in_sentence:
                        gap = t.sentence_pause_ms / 1000
                    elif ci == 0:
                        gap = t.phrase_pause_ms / 1000
                    else:
                        gap = 0.06
                    jobs.append((chunk, gap))
                    first_in_sentence = False
        return jobs

    # ------------------------------------------------------------------ acoustic model
    def _flow(self, seq, emb, off, noise, cache):
        n = seq.shape[1] + emb.shape[1]
        if off + n > self.CAP:
            raise RuntimeError("TTS chunk too long for the model context")
        lat, eos, kv = self.s_flow.run(None, {"sequence": seq, "text_emb": emb,
                                              "offset": np.array(off, dtype=np.int64),
                                              "noise": noise, "cache": cache})
        kv = kv.reshape(self.L, 2, -1, self.H, self.D)
        cache[:, :, off:off + kv.shape[2]] = kv
        return lat, eos, off + kv.shape[2]

    def _decode(self, lat, st, doff) -> np.ndarray:
        feeds = {"latent": (lat.reshape(1, 1, self.ldim) * self.emb_std + self.emb_mean).astype(np.float32)}
        for i in range(len(st)):
            feeds[self.dec_in[1 + i]] = st[i]
        res = self.s_dec.run(None, feeds)
        for j, o in enumerate(self.out_small):
            st[self.small_slots[j]] = res[o]
        for j, o in enumerate(self.out_kv):
            st[self.cache_slots[j]][:, :, doff:doff + self.spl] = res[o]
        return res[0][0, 0]

    def _raw_blocks(self, chunk: str, rng, cancel: threading.Event) -> Iterator[Tuple[np.ndarray, bool]]:
        """(audio block, eos flagged) per latent; decoding runs in parallel with the next flow step."""
        with self._lock:
            vcache, voff = self.voice
        cache = vcache.copy()
        tokens = self.sp.encode(chunk, out_type=int)
        emb = self.lut[np.asarray(tokens)][None].astype(np.float32)
        temp = self.config.tts.temperature
        noise = lambda: (rng.standard_normal((1, self.ldim)) * temp ** 0.5).astype(np.float32)
        lat, eos, off = self._flow(np.full((1, 1, self.ldim), np.nan, np.float32), emb, voff, noise(), cache)
        st = [a.copy() for a in self.dec_init]
        max_len = int(np.ceil((len(tokens) / self.tps + self.pad_s) * self.frame_rate))
        pending = self._pool.submit(self._decode, lat, st, 0)
        eos_seen = False
        for step in range(max_len):
            if cancel.is_set():
                pending.cancel()
                return
            nxt, eos, off = self._flow(lat.reshape(1, 1, self.ldim), np.zeros((1, 0, self.dim), np.float32),
                                       off, noise(), cache)
            blk = pending.result()
            eos_seen = eos_seen or bool(eos.reshape(-1)[0])
            pending = self._pool.submit(self._decode, nxt, st, (step + 1) * self.spl)
            lat = nxt
            if (yield blk, eos_seen):          # caller sends True to stop
                break
        yield pending.result(), True           # one more block: the fade-out tail

    def _chunk_audio(self, chunk: str, rng, cancel: threading.Event) -> Iterator[np.ndarray]:
        """Clean streaming: trim leading silence, squeeze dead air, stop at the real end, even loudness."""
        t = self.config.tts
        words = len(chunk.split())
        after_eos = (3 if words <= 4 else 1) + 2
        spoke, quiet, since_eos, fade_in = False, 0, 0, True
        gen = self._raw_blocks(chunk, rng, cancel)
        stop = False
        try:
            blk, eos = next(gen)
            while True:
                r = float(np.sqrt(np.mean(blk * blk)))
                if eos:
                    since_eos += 1
                tail = stop
                if r >= self.SPOKEN:
                    spoke, quiet = True, 0
                    self._level = 0.9 * self._level + 0.1 * r
                elif spoke:
                    quiet += 1
                emit = spoke and (quiet <= 3 or tail)      # skip leading silence and long dead air
                if emit:
                    out = blk
                    if t.loudness:
                        out = out * float(np.clip(t.loudness / max(self._level, 1e-3), 0.5, 3.0))
                    peak = float(np.max(np.abs(out))) if len(out) else 0.0
                    if peak > 0.97:
                        out = out * (0.97 / peak)
                    if fade_in:
                        n = min(len(out), 120)
                        out = out.copy()
                        out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                        fade_in = False
                    if tail:
                        out = out * np.linspace(1.0, 0.0, len(out), dtype=np.float32)
                    yield out.astype(np.float32, copy=False)
                if tail:
                    return
                stop = ((since_eos > after_eos and r < self.SPOKEN) or since_eos > after_eos + 4
                        or (spoke and quiet >= 12) or (eos and not spoke and since_eos > after_eos + 6))
                blk, eos = gen.send(stop)
        except StopIteration:
            return
        finally:
            gen.close()

    def stream(self, plan: List[Tuple[str, float]], cancel: threading.Event) -> Iterator[np.ndarray]:
        retries = max(0, self.config.tts.retries)
        for chunk, gap in plan:
            if cancel.is_set():
                return
            if gap > 0:
                yield np.zeros(int(gap * self.sample_rate), np.float32)
            for _ in range(retries + 1):
                seed = int(self.rng.integers(0, 2 ** 63))
                got = False
                for blk in self._chunk_audio(chunk, np.random.default_rng(seed), cancel):
                    if cancel.is_set():
                        return
                    got = True
                    yield blk
                if got:
                    break

    def warmup(self) -> None:
        for _ in self.stream(self.prepare("سلام."), threading.Event()):
            pass

    def close(self) -> None:
        self._pool.shutdown(wait=False)


# =========================================================================== sherpa-onnx
@register("tts", "sherpa_onnx", "sherpa", "piper")
class SherpaOnnxTTS(TTSBackend):
    """Piper / VITS / Matcha / Kokoro / Kitten voices via sherpa-onnx.

    ``tts.model`` is a Hugging Face repo, a local folder, or a local ``.onnx`` file whose folder
    holds the rest (``tokens.txt``, ``voices.bin``, ``espeak-ng-data``).
    """

    def load(self, ctx: LoadContext) -> None:
        try:
            import sherpa_onnx  # type: ignore
        except ImportError as e:
            raise RuntimeError("this TTS needs `pip install sherpa-onnx`") from e
        from . import models
        from .lang import PIPER_VOICES
        cfg = self.config
        ref = cfg.tts.model or PIPER_VOICES.get(cfg.language, models.DEFAULT_SHERPA_TTS)
        main = ""
        if ref.endswith(".onnx") and os.path.isfile(ref):             # a single model file: its folder
            d, main = os.path.split(os.path.abspath(ref))              # holds voices.bin / tokens.txt ...
        else:
            d = models.resolve_snapshot(cfg, "tts", ref, ctx.progress)
        files = os.listdir(d)
        onnx = [main] if main else sorted((f for f in files if f.endswith(".onnx")),
                                          key=lambda f: (".int8." in f, len(f)))
        pick = lambda *names: next((os.path.join(d, n) for n in names if n in files), "")
        data_dir = os.path.join(d, "espeak-ng-data") if "espeak-ng-data" in files else ""
        tokens, lexicon = pick("tokens.txt"), pick("lexicon.txt")
        threads = cfg.tts.threads or max(1, ctx.plan.tts_threads)
        provider = "cuda" if "CUDAExecutionProvider" in ctx.plan.ort_providers else "cpu"
        M = sherpa_onnx
        if "voices.bin" in files and "kitten" in (onnx[0] + ref).lower():   # KittenTTS
            model = M.OfflineTtsModelConfig(kitten=M.OfflineTtsKittenModelConfig(
                model=os.path.join(d, onnx[0]), voices=pick("voices.bin"), tokens=tokens,
                data_dir=data_dir), num_threads=threads, provider=provider)
        elif "voices.bin" in files:                                  # Kokoro
            model = M.OfflineTtsModelConfig(kokoro=M.OfflineTtsKokoroModelConfig(
                model=os.path.join(d, onnx[0]), voices=pick("voices.bin"), tokens=tokens,
                data_dir=data_dir, lexicon=",".join(os.path.join(d, f) for f in files
                                                    if f.startswith("lexicon") and f.endswith(".txt"))),
                num_threads=threads, provider=provider)
        elif any("vocos" in f for f in files):                        # Matcha (+ vocoder)
            voc = next(os.path.join(d, f) for f in files if "vocos" in f)
            am = next(os.path.join(d, f) for f in onnx if "vocos" not in f)
            model = M.OfflineTtsModelConfig(matcha=M.OfflineTtsMatchaModelConfig(
                acoustic_model=am, vocoder=voc, tokens=tokens, lexicon=lexicon, data_dir=data_dir),
                num_threads=threads, provider=provider)
        else:                                                         # VITS / Piper
            model = M.OfflineTtsModelConfig(vits=M.OfflineTtsVitsModelConfig(
                model=os.path.join(d, onnx[0]), tokens=tokens, lexicon=lexicon, data_dir=data_dir),
                num_threads=threads, provider=provider)
        self.tts = M.OfflineTts(M.OfflineTtsConfig(model=model, max_num_sentences=1))
        self.sample_rate = int(self.tts.sample_rate)
        self.n_speakers = int(getattr(self.tts, "num_speakers", 1) or 1)

    def voices(self) -> List[str]:
        return [str(i) for i in range(self.n_speakers)]

    def stream(self, plan: str, cancel: threading.Event) -> Iterator[np.ndarray]:
        t = self.config.tts
        out = self.tts.generate(plan, sid=int(t.speaker_id), speed=float(t.speed))
        a = np.asarray(out.samples, np.float32)
        if len(a) and not cancel.is_set():
            yield a

    def warmup(self) -> None:
        self.synthesize("Hi.")


# =========================================================================== KittenTTS
@register("tts", "kitten", "kitten_tts")
class KittenTTS(TTSBackend):
    """KittenTTS (English) in its original format: ``tts.model`` is the ``.onnx`` file, with
    ``voices.npz`` (and optionally ``config.json``) beside it. ``tts.voice`` is a voice name or
    alias (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo). Needs
    ``pip install phonemizer-fork espeakng-loader``."""
    sample_rate = 24000
    _SYMBOLS = (["$"] + list(';:,.!?¡¿—…"«»“” ')
                + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
                + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"))

    def load(self, ctx: LoadContext) -> None:
        import onnxruntime as ort
        try:
            import espeakng_loader  # type: ignore
            from phonemizer.backend import EspeakBackend  # type: ignore
            from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
        except ImportError as e:
            raise RuntimeError("KittenTTS needs `pip install phonemizer-fork espeakng-loader`") from e
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        t = self.config.tts
        path = os.path.abspath(t.model)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"tts.model must be the KittenTTS .onnx file, got {t.model!r}")
        d = os.path.dirname(path)
        meta = {}
        if os.path.exists(os.path.join(d, "config.json")):
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                meta = json.load(f)
        npz = np.load(os.path.join(d, meta.get("voices", "voices.npz")))
        self._voices = {k: npz[k].astype(np.float32) for k in npz.files}
        self._aliases = {k.lower(): v for k, v in meta.get("voice_aliases", {}).items()}
        self.set_voice(t.voice)
        self._ids = {c: i for i, c in enumerate(self._SYMBOLS)}
        self._g2p = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads = t.threads or max(1, ctx.plan.tts_threads)
        self.sess = ort.InferenceSession(path, so, providers=ctx.plan.ort_providers or None)

    def set_voice(self, ref, sr=None) -> None:
        if not isinstance(ref, str):
            raise VoiceCloningNotSupported("KittenTTS cannot clone voices")
        name = self._aliases.get(ref.lower(), ref)
        self.voice = name if name in self._voices else next(iter(self._voices))

    def voices(self) -> List[str]:
        return list(self._aliases) + list(self._voices)

    def prepare(self, text: str) -> Any:
        ph = self._g2p.phonemize([text])[0]
        ph = " ".join(re.findall(r"\w+|[^\w\s]", ph))
        ids = [0] + [self._ids[c] for c in ph if c in self._ids] + [10, 0]
        return text, np.array([ids], np.int64)

    def stream(self, plan: Any, cancel: threading.Event) -> Iterator[np.ndarray]:
        text, ids = plan
        style = self._voices[self.voice]
        style = style[min(len(text), len(style) - 1)][None]
        speed = np.array([float(self.config.tts.speed)], np.float32)
        wav = self.sess.run(None, {"input_ids": ids, "style": style, "speed": speed})[0]
        wav = np.asarray(wav, np.float32).reshape(-1)[:-5000]
        if len(wav) and not cancel.is_set():
            yield wav


# =========================================================================== mock
@register("tts", "mock")
class MockTTS(TTSBackend):
    """Soft two-tone hum whose length follows the text (tests, UI demos)."""
    supports_streaming = True

    def __init__(self, config=None, sec_per_char: float = 0.05, compute_factor: float = 0.1, sr: int = 24000):
        super().__init__(config)
        self.spc, self.cf, self.sample_rate = sec_per_char, compute_factor, sr
        self.calls: List[str] = []
        self.voice: Optional[str] = None

    def stream(self, plan: str, cancel: threading.Event) -> Iterator[np.ndarray]:
        self.calls.append(plan)
        dur = max(0.2, self.spc * len(plan))
        t = np.arange(int(dur * self.sample_rate)) / self.sample_rate
        env = np.minimum(1, np.minimum(t, dur - t) * 30)
        audio = (0.1 * env * np.sin(2 * np.pi * 190 * t)).astype(np.float32)
        step = int(0.08 * self.sample_rate)
        for i in range(0, len(audio), step):
            if cancel.is_set():
                return
            time.sleep(0.08 * self.cf)
            yield audio[i:i + step]
