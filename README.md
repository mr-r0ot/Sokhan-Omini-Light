# سخن (Sokhan)

A **CPU-first, realtime, "omni-style" Persian voice engine.** Three specialized
models — a Persian streaming-friendly STT, a small multilingual LLM, and a
Persian TTS with voice cloning — fused into one tightly-integrated engine with
professional VAD, barge-in, prosody sensing, tool calling and optional
vision, so it *behaves* like a realtime omni model even though no single
"omni" checkpoint is used.

```
mic ─► VAD/endpointer ─► STT ─┐
         │  ▲                 ├─► prosody cues ─► LLM (stream) ─► chunker ─► TTS ─► speaker
   barge-in detector ◄────────┘                        ▲                              │
         └──── cancel + flush ─────────────────────────┴──────────────────────────────┘
```

Everything runs on a **plain CPU** by default. GPU is a single config toggle
(`hardware.use_gpu = True`) that auto-detects CUDA/Metal/DirectML/ROCm and
falls back to CPU cleanly if nothing is found.

## Why three models, fused

No open, Persian-fluent, CPU-runnable "omni" (speech-to-speech) model exists
today (see the design notes at the end). So instead of a loose pipeline of
scripts, `sokhan` fuses three specialized, swappable models behind **one**
event-driven engine that handles the parts that make a pipeline *feel* like a
single realtime model: adaptive turn-taking, streaming generation all the way
through, mid-sentence barge-in, tone-aware prompting, and tool calling.

| Stage | Default model | Why |
|---|---|---|
| VAD | Silero VAD (ONNX) | tiny (~2 MB), robust, runs on any CPU |
| STT | Shenava-Rizeh (32M, Persian FastConformer-CTC) | fast + accurate Persian ASR, CPU-only |
| LLM | Qwen3.5-4B, Q4_K_M quantized (GGUF) | small, multilingual, tool-calling, optional vision |
| TTS | pocket-tts-farsi-v2 (ONNX path) | Persian voice cloning, fully offline, CPU |

Every one of these is swappable — see **Configuration** below.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Linux users running `example.py` also need system Tkinter:
`sudo apt install python3-tk` (Debian/Ubuntu) or `sudo dnf install python3-tkinter` (Fedora).
Windows/macOS python.org installers already include it.

Models are **downloaded automatically on first run** (a few GB, see the
report at the bottom) and cached under `~/.cache/sokhan` (override with
`SOKHAN_HOME` or `Config.cache_dir`). After that everything works fully
offline.

## Quick start

```python
from sokhan import Config, OmniEngine
from sokhan.audio_io import LocalAudio

cfg = Config()                       # sane defaults, CPU, Persian, quantized
cfg.prompt.greeting = "سلام! چطور می‌تونم کمکتون کنم؟"

engine = OmniEngine(cfg)
engine.start(block=True)             # downloads + loads models, blocks until ready

audio = LocalAudio(engine)           # microphone in, speakers out
audio.start()

engine.run_forever()                 # Ctrl+C to stop
```

Or try the ready-made desktop app:

```bash
python example.py
```

## Ordinary use as a library (no audio hardware needed)

```python
from sokhan import Config, OmniEngine, Tool

def add_item(name: str, qty: int = 1) -> str:
    """Add a food item to the order."""
    return f"{qty}x {name} added."

engine = OmniEngine(Config(), tools=[Tool.from_function(add_item)])
engine.start(block=True)
engine.on("assistant_text", lambda text: print(text, end=""))
engine.send_text("یک پیتزا میخوام")   # or feed real mic audio via engine.feed_audio(pcm, sr)
```

`OmniEngine` is transport-agnostic: feed it PCM from a microphone, a SIP/RTP
call (Pipecat/LiveKit), a WebSocket, or a file — and consume `audio_out`
events however you like (speakers, a call leg, a wav file).

## Configuration

Every knob has a tuned default; override only what you need.

```python
from sokhan import Config

cfg = Config.preset("lowmem")        # "balanced" (default) | "lowmem" | "lowlatency" | "quality"
cfg.llm.repo_id = "unsloth/Qwen3.5-9B-GGUF"   # swap the LLM
cfg.llm.filename = "*Q4_K_M.gguf"
cfg.stt.model = "koochik"            # bigger/more accurate STT
cfg.tts.voice = "female_narration"   # built-in reference voice
cfg.hardware.use_gpu = True          # opt-in GPU, auto-detected, auto-fallback to CPU
cfg.vision.enabled = True            # allow image turns (needs a vision-capable LLM + backend)
cfg.llm.backend = "llama_server"     # spawns llama-server: needed for vision, or a GPU build
cfg.save("my_config.json")           # ship this to users; Config.load(path) reads it back
```

Anything not overridden keeps the default the engine ships with, chosen for
the best balance of realtime latency, accuracy and CPU/RAM use.

### Swapping backends entirely

Pass your own backend objects (or point `llm.backend` at any OpenAI-compatible
server — vLLM, Ollama, a cloud API) — the engine only depends on the small
`STTBackend` / `LLMBackend` / `TTSBackend` interfaces in `sokhan.stt` /
`sokhan.llm` / `sokhan.tts`, so any conforming class works, in any project.

```python
cfg.llm.backend = "openai"
cfg.llm.base_url = "http://localhost:11434/v1"   # e.g. Ollama, vLLM, LM Studio
cfg.llm.model_name = "qwen3.5:9b"
```

## Reliability & performance defaults

- **Memory guard**: refuses to load (with a clear message) if free RAM can't
  cover the configured models, instead of letting the OS thrash or the kernel
  OOM-kill the process. Disable with `hardware.memory_guard = False`.
- **CPU budgeting**: reserves cores for audio capture/playback so the UI and
  microphone never starve even while the LLM is generating.
- **Quantized by default**: 4-bit LLM. Turn off with `llm.quantized = False`
  to use a full-precision checkpoint if you have the RAM/CPU to spare.
- **Adaptive endpointing**: starts transcribing as soon as the user's turn
  *looks* finished (grammar + intonation), falling back to a fixed silence
  timeout — this is most of the perceived "realtime" feeling.
- **Streaming everywhere**: LLM tokens are chunked into speakable phrases and
  sent to TTS before the full reply is even generated.
- **Barge-in**: interrupting the agent mid-sentence cancels LLM+TTS instantly
  and keeps only what was actually spoken in the conversation history (not
  the whole unfinished reply), so the model doesn't get confused later.
- **Phrase cache**: fixed phrases (fillers, greetings) are cached after first
  synthesis — effectively zero latency.

## Cross-platform

Pure Python + numpy/scipy + onnxruntime + llama-cpp-python + sherpa-onnx +
sounddevice — all of which ship prebuilt wheels for Windows, macOS (Intel &
Apple Silicon) and Linux (x86_64 & ARM64). No platform-specific code paths.

## Reusing `sokhan` in another project

The package has no dependency on `example.py` or any particular audio stack:

```python
import sokhan
engine = sokhan.OmniEngine(sokhan.Config())
```

Drop the `sokhan/` folder into any project, or `pip install -e .` this repo,
and use it exactly like any other library.

## Licensing note (important)

The default TTS model (`mehdi-hf/pocket-tts-farsi-v2`, used via
`nimaone/persian_tts`) is released under **CC-BY-NC-4.0** — non-commercial
use only, inherited from its training data. If you deploy this commercially,
either obtain a commercial license from the model's author or swap in a
different Persian TTS backend (`cfg.tts.backend`).

## Disk / RAM / CPU report

Run:

```bash
python -m sokhan.report
```

to print your actual hardware, resource plan, and installed model sizes.
Approximate figures with all default (quantized) settings:

| Component | Disk | RAM while loaded |
|---|---:|---:|
| VAD (Silero) | ~2 MB | ~30 MB |
| STT (Shenava-Rizeh, 32M) | ~130 MB | ~350 MB |
| LLM (Qwen3.5-4B, Q4_K_M) | ~2.8 GB | ~3.3 GB |
| TTS (pocket-tts-farsi-v2, ONNX) | ~480 MB | ~700 MB |
| **Total** | **~3.4 GB** | **~4.4 GB peak, ~3.3 GB steady** |

**Minimum to run comfortably:** 2 physical CPU cores (4+ recommended for
sub-1.5s replies), ~4 GB free RAM, ~3.5 GB free disk, no GPU required.
Unquantized LLM roughly triples LLM disk/RAM (~8-9 GB total).
