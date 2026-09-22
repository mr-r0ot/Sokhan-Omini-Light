"""Configuration: plain dataclasses, JSON/YAML in, JSON out.

Every default here is tuned for "CPU only, 4-8 cores, 8 GB RAM, Persian".
Users can override anything via ``Config.load(path)`` or by editing fields.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List, Optional, Union

DEFAULT_SYSTEM_PROMPT = """You are "{name}", a friendly real-time voice assistant. The user talks to you by voice and your reply is turned into speech immediately.

# Language and voice style
- Always answer in Persian (Farsi), in natural spoken style (polite but conversational). If the user clearly speaks another language, match it.
- Keep it SHORT: usually one or two short sentences (under about 25 words). Give more only when the user explicitly asks for detail.
- Speak, don't write: no markdown, no lists, no emojis, no headings, no URLs, no parentheses, no stage directions. Numbers may be written as digits; they are read aloud automatically.
- Start with the answer itself. No filler such as "of course!" or "great question", and do not repeat the question back.
- Speech recognition can make mistakes: infer the most plausible meaning from context. If the request is genuinely unclear, ask ONE short clarifying question instead of guessing.

# Voice cues
Sometimes a note like [voice cues: high energy, fast pace, rising intonation] follows the user's words. It describes HOW the user sounded, measured from the audio. Use it only as a soft hint: match their energy, stay calm and brief if they sound rushed or upset, be warmer if they sound tired or sad. Never mention, quote, or read it aloud, and never claim to be sure about their feelings.

# Images
If an image is attached, use it when relevant and describe what matters in a few speech-friendly words.

# Honesty
If you do not know something, say so briefly. Never invent facts, prices, or personal data.
{extra}""".strip()


@dataclass
class HardwareConfig:
    use_gpu: bool = False            # user toggle; when True the engine auto-detects a GPU and falls back to CPU
    cpu_threads: int = 0             # 0 = automatic plan from physical cores
    memory_guard: bool = True        # refuse/warn before loading models that will not fit in free RAM
    min_free_ram_mb: int = 500       # safety margin that must stay free after loading
    lower_worker_priority: bool = True


@dataclass
class AudioConfig:
    input_device: Optional[Union[int, str]] = None
    output_device: Optional[Union[int, str]] = None
    input_sr: int = 16000
    output_sr: int = 24000           # TTS native rate; resampled if the device refuses it
    frame_ms: int = 32               # 512 samples @16k (Silero window)
    output_latency: str = "low"


@dataclass
class VADConfig:
    backend: str = "silero"          # "silero" | "energy"
    model_path: str = ""             # empty -> auto-download silero_vad.onnx (~2 MB)
    threshold_on: float = 0.55
    threshold_off: float = 0.35
    smoothing: float = 0.30          # EMA weight of the newest frame probability
    min_speech_ms: int = 160         # ignore clicks/coughs shorter than this
    pre_roll_ms: int = 320           # audio kept before speech onset
    fast_silence_ms: int = 340       # earliest moment we may end a turn (if text looks complete)
    silence_ms: int = 760            # hard end-of-turn silence
    trailing_pad_ms: int = 160
    max_utterance_s: float = 25.0
    noise_gate_db: float = 7.0       # speech must be this far above the adaptive noise floor
    abs_min_db: float = -58.0
    adaptive_endpointing: bool = True
    early_complete_threshold: float = 0.9   # end the turn early if the transcript looks this complete


@dataclass
class BargeInConfig:
    enabled: bool = True
    threshold: float = 0.80          # stricter than normal VAD while the agent talks
    min_speech_ms: int = 300
    echo_margin_db: float = 4.0      # mic must exceed the playback reference by this many dB
    half_duplex: bool = False        # True = ignore the mic while speaking (best on open speakers)
    echo_tail_ms: int = 220          # ignore mic briefly after playback ends (room echo)


@dataclass
class STTConfig:
    backend: str = "sherpa_nemo_ctc"  # or "mock"
    model: str = "rizeh"             # "rizeh" (32M, fast) | "koochik" (114M, most accurate) | custom dir
    model_dir: str = ""              # local dir with model.onnx + tokens.txt (skips download)
    repo_id: str = ""                # override HF repo
    num_threads: int = 0
    itn: bool = True                 # spoken numbers -> digits
    auto_gain: bool = True
    target_dbfs: float = -23.0
    partials: bool = True            # live partial transcripts while the user speaks
    partial_interval_ms: int = 500
    min_audio_ms: int = 180


@dataclass
class LLMConfig:
    backend: str = "llama_cpp"       # "llama_cpp" (in-process) | "llama_server" | "openai" (any compatible URL)
    repo_id: str = "unsloth/Qwen3.5-4B-GGUF"
    filename: str = "*Q4_K_M.gguf"   # quantized default
    unquantized_filename: str = "*BF16.gguf"
    quantized: bool = True           # False -> use unquantized_filename
    model_path: str = ""             # local .gguf (skips download)
    mmproj_filename: str = "*mmproj*F16*.gguf"
    mmproj_path: str = ""
    n_ctx: int = 4096
    n_batch: int = 256
    n_threads: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    flash_attn: bool = False
    max_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repeat_penalty: float = 1.05
    presence_penalty: float = 0.0
    enable_thinking: bool = False    # Qwen3.5 thinks by default; that is far too slow for voice
    history_max_turns: int = 10
    keep_empty_think_in_history: bool = True   # keeps the KV-cache prefix identical -> big prefill saving
    # openai-compatible / llama-server
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = ""
    model_name: str = "default"
    server_binary: str = ""          # path to llama-server; empty -> search PATH
    server_port: int = 0             # 0 -> pick a free port
    server_extra_args: List[str] = field(default_factory=list)
    request_timeout_s: float = 120.0
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSConfig:
    backend: str = "parsigo_onnx"    # or "mock"
    engine_dir: str = ""             # checkout of nimaone/persian_tts (auto-downloaded when empty)
    voice: str = ""                  # reference wav (<= 5 s). empty -> bundled male_hello.wav
    mode: str = "pack"               # "pack" (fluent) | "split" (pause at commas)
    pace: float = 1.0
    seed: Optional[int] = None
    cache_phrases: bool = True
    cache_max_mb: int = 96
    first_chunk_min_chars: int = 14
    chunk_min_chars: int = 30
    chunk_max_chars: int = 150


@dataclass
class ProsodyConfig:
    enabled: bool = True
    inject_into_prompt: bool = True
    min_confidence: float = 0.45
    min_utterance_ms: int = 600
    baseline_alpha: float = 0.25
    language: str = "en"             # language of the cue words injected into the prompt


@dataclass
class VisionConfig:
    enabled: bool = False            # off by default; the UI/user can switch it on
    max_side: int = 448              # downscale to keep visual tokens (and latency) small
    jpeg_quality: int = 80


@dataclass
class PromptConfig:
    name: str = "سخن"
    system: str = ""                 # empty -> DEFAULT_SYSTEM_PROMPT
    extra: str = ""                  # appended to the default prompt (business rules etc.)
    greeting: str = ""               # spoken once when ready ("" = silent start)
    fillers: List[str] = field(default_factory=lambda: ["یک لحظه.", "لحظه‌ای صبر کنید."])
    filler_after_ms: int = 1300      # speak a filler if the model is still silent after this long


@dataclass
class Config:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    cache_dir: str = ""              # empty -> $SOKHAN_HOME or ~/.cache/sokhan

    # ---- helpers -------------------------------------------------------
    def system_prompt(self) -> str:
        p = self.prompt
        if p.system.strip():
            return p.system.replace("{name}", p.name).replace("{extra}", p.extra)
        extra = ("\n# Extra instructions\n" + p.extra.strip()) if p.extra.strip() else ""
        return DEFAULT_SYSTEM_PROMPT.replace("{name}", p.name).replace("{extra}", extra)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls()
        _update(cfg, data)
        return cfg

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yml", ".yaml")):
            import yaml  # optional
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def preset(cls, name: str) -> "Config":
        """Ready-made profiles. ``balanced`` == defaults."""
        cfg = cls()
        name = (name or "balanced").lower()
        if name == "lowmem":            # ~3.3 GB RAM total
            cfg.llm.n_ctx = 2048
            cfg.llm.history_max_turns = 6
            cfg.stt.model = "rizeh"
            cfg.stt.partials = False
            cfg.tts.cache_max_mb = 32
        elif name == "lowlatency":
            cfg.vad.fast_silence_ms = 280
            cfg.vad.silence_ms = 620
            cfg.llm.max_tokens = 110
            cfg.tts.chunk_min_chars = 22
            cfg.tts.first_chunk_min_chars = 10
        elif name == "quality":
            cfg.stt.model = "koochik"
            cfg.llm.max_tokens = 260
            cfg.llm.n_ctx = 6144
            cfg.vad.silence_ms = 900
        elif name != "balanced":
            raise ValueError(f"unknown preset {name!r}")
        return cfg

    def copy(self) -> "Config":
        return copy.deepcopy(self)


def _update(obj: Any, data: Dict[str, Any]) -> None:
    names = {f.name: f for f in fields(obj)}
    for key, val in (data or {}).items():
        if key not in names:
            continue  # ignore unknown keys: forward/backward compatible configs
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _update(cur, val)
        else:
            setattr(obj, key, val)


def cache_root(cfg: Config) -> str:
    root = cfg.cache_dir or os.environ.get("SOKHAN_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "sokhan")
    os.makedirs(root, exist_ok=True)
    return root
