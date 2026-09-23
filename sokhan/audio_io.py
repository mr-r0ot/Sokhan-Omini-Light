"""Deprecated alias of :mod:`sokhan.audio` (kept so old imports keep working)."""
from .audio import (LocalAudio, PlaybackBuffer, list_devices, read_audio, record, resample,
                    to_float32, to_pcm16, write_wav)

__all__ = ["LocalAudio", "PlaybackBuffer", "list_devices", "read_audio", "record", "resample",
           "to_float32", "to_pcm16", "write_wav"]
