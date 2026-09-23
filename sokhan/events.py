"""A tiny thread-safe event bus.

Handlers run on the engine's worker threads. Keep them quick, and hand work
over to your own thread (or UI loop) when it is slow. A handler that raises
is logged and never breaks the engine.

    omni.on("transcript", lambda text: print("you:", text))

    @omni.on("response_done")
    def done(text, interrupted):
        ...

    omni.on("*", lambda event, **data: ...)      # every event (bridges, loggers)

Handlers may accept only the keyword arguments they care about.
"""
from __future__ import annotations

import inspect
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("sokhan.events")

#: Every event the engine emits, with its keyword arguments.
EVENTS: Dict[str, str] = {
    "ready": "models loaded, engine running",
    "load_progress": "stage, fraction, message",
    "state": "state (State: idle | listening | thinking | speaking)",
    "speech_start": "user started talking",
    "speech_end": "duration_ms",
    "partial_transcript": "text (live, while the user speaks)",
    "transcript": "text (final user utterance)",
    "response_start": "turn_id",
    "response_delta": "text (streamed LLM text, speakable part only)",
    "response_done": "text, interrupted",
    "tool_call": "name, arguments, result",
    "audio": "audio (float32 numpy), sr",
    "audio_flush": "playback must stop now (interruption)",
    "interrupted": "reason, heard (text the user actually heard)",
    "level": "db, speech (0..1 VAD probability)",
    "prosody": "features, cue",
    "metrics": "latency_ms and per-stage timings of the last turn",
    "warning": "message",
    "error": "stage, error",
}


class _Handler:
    __slots__ = ("fn", "names", "var_kw", "once")

    def __init__(self, fn: Callable, once: bool):
        self.fn, self.once = fn, once
        try:
            params = inspect.signature(fn).parameters.values()
            self.var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
            self.names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        except (TypeError, ValueError):          # builtins / C callables: pass everything
            self.var_kw, self.names = True, set()

    def __call__(self, kw: Dict[str, Any]) -> None:
        if self.var_kw:
            self.fn(**kw)
        else:
            self.fn(**{k: v for k, v in kw.items() if k in self.names})


class EventBus:
    def __init__(self) -> None:
        self._h: Dict[str, List[_Handler]] = {}
        self._lock = threading.Lock()

    def on(self, event: str, fn: Optional[Callable] = None, *, once: bool = False):
        """Subscribe. Usable directly or as a decorator. Returns ``fn`` (decorator form) or
        an ``unsubscribe()`` callable."""
        if fn is None:
            def deco(f: Callable) -> Callable:
                self.on(event, f, once=once)
                return f
            return deco
        if event != "*" and event not in EVENTS:
            log.warning("subscribing to unknown event %r (known: %s)", event, ", ".join(EVENTS))
        h = _Handler(fn, once)
        with self._lock:
            self._h.setdefault(event, []).append(h)
        return lambda: self._remove(event, h)

    def once(self, event: str, fn: Callable):
        return self.on(event, fn, once=True)

    def off(self, event: str, fn: Optional[Callable] = None) -> None:
        with self._lock:
            if fn is None:
                self._h.pop(event, None)
            else:
                self._h[event] = [h for h in self._h.get(event, []) if h.fn is not fn]

    def _remove(self, event: str, h: _Handler) -> None:
        with self._lock:
            lst = self._h.get(event, [])
            if h in lst:
                lst.remove(h)

    def emit(self, event: str, **kw: Any) -> None:
        with self._lock:
            specific = list(self._h.get(event, ()))
            wild = list(self._h.get("*", ()))
        for h in specific:
            self._call(event, h, kw)
        if wild:
            tagged = dict(kw, event=event)
            for h in wild:
                self._call("*", h, tagged)

    def _call(self, key: str, h: _Handler, kw: Dict[str, Any]) -> None:
        try:
            if h.once:
                self._remove(key, h)
            h(kw)
        except Exception:
            log.exception("handler for %r failed", kw.get("event", key))

    def has(self, event: str) -> bool:
        return bool(self._h.get(event)) or bool(self._h.get("*"))
