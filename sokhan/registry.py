"""Backend registry: how config strings become model objects.

Every stage (``vad``, ``stt``, ``llm``, ``tts``) picks its implementation from
``config.<stage>.backend``:

* a built-in name        ``cfg.tts.backend = "sherpa_onnx"``
* your own registration  ``@register("tts", "my_tts") class MyTTS(TTSBackend): ...``
* an import path         ``cfg.tts.backend = "my_package.voices:MyTTS"``

Or skip the config entirely and pass an instance: ``Omni(cfg, tts=MyTTS())``.
Backends receive the full ``Config`` in their constructor.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type


@dataclass
class LoadContext:
    """Passed to every backend's ``load()``."""
    config: Any                                   # sokhan.Config
    plan: Any                                     # sokhan.hardware.ResourcePlan
    progress: Optional[Callable[[str, float, str], None]] = None

    def report(self, stage: str, fraction: float, message: str = "") -> None:
        if self.progress:
            self.progress(stage, fraction, message)

_BUILTIN_MODULES = {"vad": "sokhan.vad", "stt": "sokhan.stt", "llm": "sokhan.llm", "tts": "sokhan.tts"}
_REGISTRY: Dict[str, Dict[str, Type]] = {k: {} for k in _BUILTIN_MODULES}


def register(kind: str, *names: str) -> Callable[[Type], Type]:
    """Class decorator: ``@register("stt", "my_stt")``."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown backend kind {kind!r}; expected one of {sorted(_REGISTRY)}")

    def deco(cls: Type) -> Type:
        for n in names:
            _REGISTRY[kind][n.lower()] = cls
        return cls
    return deco


def available(kind: str) -> Dict[str, Type]:
    importlib.import_module(_BUILTIN_MODULES[kind])
    return dict(_REGISTRY[kind])


def resolve(kind: str, name: str) -> Type:
    if ":" in name:                                   # "package.module:ClassName"
        mod, _, attr = name.partition(":")
        return getattr(importlib.import_module(mod), attr)
    key = name.lower()
    if key not in _REGISTRY[kind]:
        importlib.import_module(_BUILTIN_MODULES[kind])
    if key not in _REGISTRY[kind]:
        raise ValueError(f"unknown {kind} backend {name!r}. Built-in: {', '.join(sorted(_REGISTRY[kind]))}. "
                         f"Register your own with sokhan.register({kind!r}, 'name') or use 'module:Class'.")
    return _REGISTRY[kind][key]


def create(kind: str, config):
    section = getattr(config, kind)
    return resolve(kind, section.backend)(config)
