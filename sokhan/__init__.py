"""Sokhan (سخن) — a CPU-first, realtime, omni-style voice engine.

STT (Shenava) + LLM (Qwen3.5) + TTS (pocket-tts-farsi) fused into one
event-driven engine with VAD, prosody sensing, barge-in and streaming.
"""
__version__ = "0.1.0"

from .config import Config  # noqa: E402
from .tools import Tool  # noqa: E402
from .engine import OmniEngine, State  # noqa: E402

__all__ = ["Config", "OmniEngine", "State", "Tool", "__version__"]
