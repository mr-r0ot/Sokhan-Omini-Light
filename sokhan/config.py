"""Configuration.

Plain dataclasses, one section per stage. Every field has a tuned default, so
``Config()`` is a complete, working setup; override only what you need::

    cfg = Config()
    cfg.llm.model = "unsloth/Qwen3.5-9B-GGUF"      # any GGUF repo or a local file
    cfg.llm.temperature = 0.6
    cfg.tts.voice = "my_voice.wav"                 # ~5 s sample -> voice cloning
    cfg.save("assistant.json")                     # Config.load(...) reads it back

Sections are also settable from dicts (``Config.from_dict`` / ``cfg.update``)
and with dotted keys (``cfg.set("llm.top_k", 40)``).
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

log = logging.getLogger("sokhan.config")

LANGUAGE_NAMES = {"fa": "Persian (Farsi)", "en": "English", "ar": "Arabic", "tr": "Turkish",
                  "de": "German", "fr": "French", "es": "Spanish"}

DEFAULT_SYSTEM_PROMPT = """You are {name}, a warm and quick real-time voice assistant. The user talks to you by voice and every word you write is spoken aloud immediately.

How to speak:
- Reply in {language}, in a natural, friendly, spoken style. If the user clearly speaks another language, answer in theirs.
- Be brief: one or two short sentences. Say more only when asked.
- Plain speech only: no markdown, lists, emojis, links or stage directions.
- Lead with the answer. Never repeat the question back.
- Speech recognition can mishear. Infer the most likely meaning, or ask one short question when it is truly unclear.
- Square-bracket notes in user messages, such as [voice: fast, excited] or [interrupted after: "..."], describe how the user sounded or what they actually heard of your last reply. Use them silently and never read them out.
- If you don't know something, say so briefly. Never invent facts.{extra}"""


# --------------------------------------------------------------------------- sections
@dataclass
class HardwareConfig:
    use_gpu: bool = False            # opt-in; auto-detects CUDA/Metal/ROCm/DirectML, falls back to CPU
    threads: int = 0                 # CPU threads budget (0 = physical cores)
    memory_guard: bool = True        # refuse to load when free RAM clearly cannot hold the models
    min_free_ram_mb: int = 400
    low_priority: bool = False       # lower the process priority (keeps a busy desktop responsive)


@dataclass
class AudioConfig:
    input_device: Optional[Union[int, str]] = None
    output_device: Optional[Union[int, str]] = None
    input_sr: int = 16000
    output_sr: int = 0               # 0 = the TTS model's native rate (resampled if the device refuses)
    block_ms: int = 20               # microphone block size
    output_latency: Union[str, float] = "low"


@dataclass
class VADConfig:
    backend: str = "silero"          # "silero" | "energy" (no model file) | "module:Class"
    model: str = ""                  # local silero_vad.onnx (empty = auto-download, ~2 MB)
    threshold: float = 0.5           # speech probability that opens a segment
    neg_threshold: float = 0.3       # ...and keeps it open
    smoothing: float = 0.35          # EMA weight of the newest frame
    min_speech_ms: int = 160         # shorter blips (clicks, coughs) are ignored
    pre_roll_ms: int = 320           # audio kept from before the detected onset
    trailing_pad_ms: int = 120
    noise_gate_db: float = 6.0       # speech must stand this far above the adaptive noise floor
    abs_min_db: float = -60.0


@dataclass
class TurnConfig:
    """Turn-taking: when the user's turn ends, and what happens when they talk over us."""
    speculative: bool = True         # start thinking in the user's first pause; the audio is held
                                     # back until the turn is confirmed and dropped if they go on
    speculate_after_ms: int = 240    # silence that triggers transcription + the speculative start
    end_silence_ms: int = 500        # confirm the end of turn when the transcript looks complete
    end_silence_max_ms: int = 1000   # ...when it looks unfinished ("and", "because", ...)
    max_utterance_s: float = 30.0
    barge_in: bool = True            # let the user interrupt the assistant mid-sentence
    barge_in_threshold: float = 0.7  # stricter than normal VAD while the assistant is talking
    barge_in_min_ms: int = 260
    echo_margin_db: float = 5.0      # without AEC the mic must beat the learned speaker->mic echo
    echo_tail_ms: int = 180          # ignore the mic briefly after playback (room reverb)
    half_duplex: bool = False        # True = never listen while speaking (open speakers, no AEC)
    interruption_note: bool = True   # tell the LLM what the user actually heard before cutting in
    filler_after_ms: int = 0         # >0: say a short filler if nothing is audible after this long


@dataclass
class STTConfig:
    backend: str = "sherpa_onnx"     # "sherpa_onnx" | "mock" | "module:Class"
    model: str = "Reza2kn/Shenava-Rizeh-v1.0-sherpa-onnx"   # HF repo id or a local directory
    model_type: str = "nemo_ctc"     # nemo_ctc | whisper | transducer | paraformer | sense_voice | zipformer_ctc
    language: str = ""               # whisper / sense_voice only ("" = config.language)
    quant: str = "q4"                # "q4" (4-bit, quantized locally once) | "int8" (if shipped) | "fp32"
    threads: int = 0                 # 0 = automatic
    itn: bool = True                 # spoken numbers -> digits
    auto_gain: bool = True
    target_dbfs: float = -23.0
    partials: bool = True            # live partial transcripts while the user speaks
    partial_interval_ms: int = 320
    min_audio_ms: int = 200


@dataclass
class LLMConfig:
    backend: str = "llama_cpp"       # "llama_cpp" (in-process) | "openai" (any compatible server)
                                     # | "llama_server" (managed llama.cpp server) | "module:Class"
    model: str = "unsloth/Qwen3.5-4B-GGUF"   # HF repo id, or a local .gguf file
    quant: str = "Q4_K_M"            # GGUF quantization picked from the repo (4-bit by default)
    mmproj: str = ""                 # vision projector (auto-picked from the repo when vision is on)
    chat_format: str = "auto"        # "auto" | "chatml" | "jinja" (the GGUF's own template)
    # ---- sampling
    max_tokens: int = 220
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = -1
    thinking: bool = False           # reasoning models: far too slow for voice, keep it off
    # ---- runtime
    n_ctx: int = 4096
    n_batch: int = 256
    threads: int = 0                 # 0 = automatic
    gpu_layers: int = -1             # used only with hardware.use_gpu (-1 = all layers)
    flash_attn: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    history_tokens: int = 1800       # history budget; older turns are compacted away in the background
    checkpoints: int = 3             # saved model states for instant rollback (hybrid/recurrent models)
    # ---- openai / llama_server
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = ""
    model_name: str = ""             # model id sent to the server ("" = server default)
    server_binary: str = ""
    server_args: List[str] = field(default_factory=list)
    timeout_s: float = 120.0
    extra: Dict[str, Any] = field(default_factory=dict)   # passed to the backend verbatim


@dataclass
class TTSConfig:
    backend: str = "pocket_tts"      # "pocket_tts" (Persian, voice cloning) | "sherpa_onnx" (VITS /
                                     # Piper / Kokoro / Matcha) | "mock" | "module:Class"
    model: str = ""                  # "" = backend default; HF repo id or local dir otherwise
    voice: str = "female_narration"  # built-in voice name, or a path to a ~5 s WAV to clone
    quant: str = "q4"                # "q4" | "fp32"
    streaming: bool = True           # play audio while it is being generated (~80 ms blocks)
    temperature: float = 0.3         # sampling noise of the acoustic model
    seed: Optional[int] = None
    speed: float = 1.0
    speaker_id: int = 0              # multi-speaker models (sherpa_onnx)
    threads: int = 0
    retries: int = 1                 # regenerate a chunk that came out silent
    sentence_pause_ms: int = 220
    phrase_pause_ms: int = 110
    loudness: float = 0.1            # target speech RMS (automatic gain); 0 = off
    first_chunk_min_chars: int = 10
    first_chunk_max_words: int = 4   # start speaking after this many words even without punctuation
    chunk_min_chars: int = 40
    chunk_max_chars: int = 160
    cache_phrases: bool = True       # remember the audio of short repeated phrases


@dataclass
class ProsodyConfig:
    enabled: bool = True             # measure pitch / energy / pace and give the LLM a hint
    inject: bool = True
    min_confidence: float = 0.45
    min_utterance_ms: int = 700
    baseline_alpha: float = 0.25


@dataclass
class VisionConfig:
    enabled: bool = False            # needs a vision LLM + projector (Qwen3.5 ships one)
    max_side: int = 448
    jpeg_quality: int = 80


@dataclass
class PromptConfig:
    name: str = "Sokhan"
    system: str = ""                 # full override ({name}, {language}, {extra} are filled in)
    extra: str = ""                  # appended to the default prompt (persona, business rules)
    greeting: str = ""               # spoken once when ready
    fillers: List[str] = field(default_factory=list)   # [] = the language pack's fillers


# --------------------------------------------------------------------------- root
@dataclass
class Config:
    language: str = "fa"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    cache_dir: str = ""              # "" = $SOKHAN_HOME or ~/.cache/sokhan

    # ------------------------------------------------------------------ prompt
    def system_prompt(self, tools: Sequence[str] = ()) -> str:
        """The system prompt; ``tools`` = names of the registered functions."""
        p = self.prompt
        lang = LANGUAGE_NAMES.get(self.language, self.language)
        extra = ("\n\n" + p.extra.strip()) if p.extra.strip() else ""
        if tools:
            extra += (f"\n\nYour functions: {', '.join(tools)}. When a request matches one, call it right away "
                      "(never claim you did something without calling it), then tell the user the result.")
        tpl = p.system.strip() or DEFAULT_SYSTEM_PROMPT
        return tpl.replace("{name}", p.name).replace("{language}", lang).replace("{extra}", extra)

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        if path.lower().endswith((".yml", ".yaml")):
            import yaml  # optional dependency
            text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        else:
            text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls().update(data)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yml", ".yaml")):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def update(self, data: Dict[str, Any]) -> "Config":
        _merge(self, data or {}, "")
        return self

    def set(self, dotted: str, value: Any) -> "Config":
        """``cfg.set("llm.temperature", 0.5)``"""
        *path, last = dotted.split(".")
        obj: Any = self
        for p in path:
            obj = getattr(obj, p)
        if not hasattr(obj, last):
            raise AttributeError(f"unknown config key {dotted!r}")
        setattr(obj, last, value)
        return self

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ presets
    @classmethod
    def for_language(cls, code: str, base: Optional["Config"] = None) -> "Config":
        """A config whose speech models fit ``code`` (ISO 639-1, e.g. "en", "de", "ar").

        Persian keeps the specialised defaults (Shenava STT + Pocket-TTS with
        voice cloning). Other languages get multilingual Whisper STT and a
        Piper voice for that language; the LLM (Qwen3.5) is multilingual already.
        """
        from .lang import MULTILINGUAL_STT, PIPER_VOICES
        cfg = (base or cls()).copy()
        code = code.lower().split("-")[0]
        cfg.language = code
        if code == "fa":
            return cfg
        cfg.stt.model, cfg.stt.model_type, cfg.stt.quant = MULTILINGUAL_STT, "whisper", "int8"
        cfg.stt.language = code
        cfg.tts.backend, cfg.tts.voice = "sherpa_onnx", ""
        cfg.tts.model = PIPER_VOICES.get(code, "")
        if not cfg.tts.model:
            log.warning("no built-in voice for %r: set cfg.tts.model to a sherpa-onnx TTS repo for it "
                        "(e.g. a 'csukuangfj/vits-piper-<locale>-<voice>' repo)", code)
            cfg.tts.model = PIPER_VOICES["en"]
        return cfg

    @classmethod
    def preset(cls, name: str = "balanced") -> "Config":
        """``balanced`` (default) | ``fast`` | ``lowmem`` | ``quality``"""
        cfg = cls()
        name = (name or "balanced").lower()
        if name == "fast":
            cfg.turn.speculate_after_ms = 200
            cfg.turn.end_silence_ms = 420
            cfg.turn.end_silence_max_ms = 850
            cfg.llm.max_tokens = 140
            cfg.prosody.enabled = False
        elif name == "lowmem":
            cfg.llm.model = "unsloth/Qwen3.5-2B-GGUF"
            cfg.llm.n_ctx = 2048
            cfg.llm.history_tokens = 900
            cfg.llm.checkpoints = 2
            cfg.stt.partials = False
        elif name == "quality":
            cfg.stt.model = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"
            cfg.llm.quant = "Q5_K_M"
            cfg.llm.max_tokens = 320
            cfg.llm.n_ctx = 6144
            cfg.llm.history_tokens = 3000
            cfg.tts.quant = "fp32"
            cfg.tts.retries = 2
            cfg.turn.end_silence_ms = 600
        elif name != "balanced":
            raise ValueError(f"unknown preset {name!r} (balanced | fast | lowmem | quality)")
        return cfg


def _merge(obj: Any, data: Dict[str, Any], prefix: str) -> None:
    names = {f.name for f in fields(obj)}
    for key, val in data.items():
        if key not in names:
            log.warning("ignoring unknown config key %r", prefix + key)
            continue
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val, prefix + key + ".")
        else:
            setattr(obj, key, val)


def cache_root(cfg: Optional[Config] = None) -> str:
    root = ((cfg.cache_dir if cfg else "") or os.environ.get("SOKHAN_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache", "sokhan"))
    os.makedirs(root, exist_ok=True)
    return root
