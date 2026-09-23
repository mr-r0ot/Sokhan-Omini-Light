"""Sokhan - a realtime, omni-style voice assistant engine made of specialist models.

    from sokhan import Omni

    omni = Omni()          # VAD + STT + LLM + TTS, 4-bit, CPU, downloaded on first run
    omni.start()
    omni.listen()          # talk to it
    omni.run_forever()

Swap any stage through ``Config`` (see ``sokhan.config``), add tools with
``@tool``, clone a voice with ``omni.clone_voice("me.wav")``.
"""
__version__ = "1.0.0"

import logging as _logging

from .config import Config  # noqa: E402
from .engine import Omni, OmniEngine, State  # noqa: E402
from .lang import LanguagePack, register_language  # noqa: E402
from .registry import register  # noqa: E402
from .tools import Tool, tool  # noqa: E402
from .tts import VoiceCloningNotSupported  # noqa: E402

_logging.getLogger("sokhan").addHandler(_logging.NullHandler())

__all__ = ["Omni", "OmniEngine", "Config", "State", "Tool", "tool", "register", "LanguagePack",
           "register_language", "VoiceCloningNotSupported", "__version__"]
