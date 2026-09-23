"""Language-neutral text utilities for a voice pipeline.

* ``clean_for_tts``   - make LLM output speakable (markdown, emoji, URLs, numbers via the language pack)
* ``SentenceChunker`` - cut a token stream into speakable chunks, the first one as early as possible
* ``StreamFilter``    - split the raw LLM stream into speech, hidden reasoning and tool calls
                        (JSON *and* Qwen3.5-style ``<function=...>`` XML calls)
* ``completeness``    - does a transcript look like a finished turn? (end-of-turn detection)
"""
from __future__ import annotations

import json
import re
from typing import Iterator, List, Optional, Tuple

from .lang import get_language

_ZW = re.compile("[​‎‏‪-‮⁦-⁩﻿]")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_MD = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s*|^\s*[-*•]\s+|^\s*\d+[.)]\s+)", re.M)
_URL = re.compile(r"https?://\S+|www\.\S+")
_BRACKETS = re.compile(r"\([^)]{0,80}\)|\[[^\]]{0,80}\]")
_SPEAKABLE = re.compile(r"\w", re.U)


def has_speakable(text: str) -> bool:
    return bool(_SPEAKABLE.search(text or ""))


def clean_for_tts(text: str, language: str = "en", spell_numbers: bool = True) -> str:
    """Strip what a voice cannot say; spell numbers out with the language pack."""
    t = _ZW.sub(" ", text)
    t = _URL.sub(" ", t)
    t = _MD.sub("", t)
    t = _BRACKETS.sub(" ", t)
    t = _EMOJI.sub("", t)
    if spell_numbers:
        t = get_language(language).normalize(t)
    t = re.sub(r"[\"“”«»<>{}|~^=+\\/*_#@]", " ", t)
    t = re.sub(r"\s*\n+\s*", ". ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"([.!?؟،,;؛:])\1+", r"\1", t)
    return t


def completeness(text: str, language: str = "en") -> float:
    return get_language(language).completeness(text)


def strip_notes(text: str) -> str:
    """Remove the engine's own bracket notes ([voice: ...], [interrupted after: ...])."""
    return re.sub(r"\s*\[(?:voice|interrupted after)[^\]]*\]\s*", " ", text).strip()


# --------------------------------------------------------------------------- chunking for TTS
_STRONG = ".!?؟…。！？\n"
_WEAK = "،,;؛:、，"


class SentenceChunker:
    """Token stream -> speakable chunks.

    The first chunk leaves at its first clause boundary (or after a handful of
    words) so audio starts early; later chunks prefer whole sentences, which
    sound better.
    """

    def __init__(self, first_min: int = 10, min_chars: int = 40, max_chars: int = 160,
                 first_max_words: int = 7):
        self.first_min, self.min_chars, self.max_chars = first_min, min_chars, max_chars
        self.first_max_words = first_max_words
        self.buf = ""
        self.emitted = 0

    def push(self, delta: str) -> List[str]:
        self.buf += delta
        return self._drain(final=False)

    def flush(self) -> List[str]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> List[str]:
        out: List[str] = []
        while True:
            cut = self._cut(final)
            if cut is None:
                break
            chunk, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if has_speakable(chunk):
                out.append(chunk)
                self.emitted += 1
        if final:
            rest, self.buf = self.buf.strip(), ""
            if has_speakable(rest):
                out.append(rest)
                self.emitted += 1
        return out

    def _cut(self, final: bool) -> Optional[int]:
        b = self.buf
        first = self.emitted == 0
        weak_at = self.first_min if first else self.min_chars + 20
        for i, ch in enumerate(b):
            if ch in _STRONG:
                if ch == "." and 0 < i < len(b) - 1 and b[i - 1].isdigit() and b[i + 1].isdigit():
                    continue                                   # decimal point
                if i + 1 >= len(b) and not final:
                    return None                                # "..." or "3." may continue
                if i + 1 >= self.first_min or ch == "\n":
                    return i + 1
            elif ch in _WEAK and i + 1 < len(b) and i + 1 >= weak_at:
                return i + 1
        if first:
            words = b.split(" ")                               # (the last piece may be unfinished)
            if len(words) > self.first_max_words:              # no punctuation yet: go anyway
                return len(" ".join(words[: self.first_max_words]))
        if len(b) > self.max_chars:
            sp = b.rfind(" ", 0, self.max_chars)
            return sp if sp > 20 else self.max_chars
        return None


# --------------------------------------------------------------------------- LLM stream filter
_OPEN = ("<think>", "<tool_call>", "<function=")
_CLOSE = {"<think>": "</think>", "<tool_call>": "</tool_call>", "<function=": "</function>"}


class StreamFilter:
    """Splits a raw LLM stream into ("text", str) and ("tool", {"name", "arguments"}).

    Hidden reasoning (``<think>``) is dropped. Tool calls are recognised both as
    JSON ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` and as the
    Qwen3.5 XML form ``<tool_call><function=name><parameter=x>1</parameter>
    </function></tool_call>``. Tags split across stream chunks are handled.
    """

    def __init__(self, start_in_think: bool = False):
        self.buf = ""
        self.mode: Optional[str] = "<think>" if start_in_think else None
        self.inner = ""

    def feed(self, delta: str) -> Iterator[Tuple[str, object]]:
        self.buf += delta
        while True:
            if self.mode is None:
                pos = self.buf.find("<")
                if pos < 0:
                    if self.buf:
                        yield "text", self.buf
                    self.buf = ""
                    return
                if pos > 0:
                    yield "text", self.buf[:pos]
                    self.buf = self.buf[pos:]
                hit = next((t for t in _OPEN if self.buf.startswith(t)), None)
                if hit:
                    self.mode, self.inner = hit, (hit if hit == "<function=" else "")
                    self.buf = self.buf[len(hit):]
                    continue
                if self.buf.startswith("</think>"):             # stray closing tag
                    self.buf = self.buf[len("</think>"):]
                    continue
                if any(t.startswith(self.buf) for t in _OPEN + ("</think>",)):
                    return                                      # may still become a tag
                yield "text", self.buf[0]
                self.buf = self.buf[1:]
            else:
                close = _CLOSE[self.mode]
                pos = self.buf.find(close)
                if pos >= 0:
                    self.inner += self.buf[:pos] + (close if self.mode == "<function=" else "")
                    self.buf = self.buf[pos + len(close):]
                    if self.mode != "<think>":
                        call = parse_tool_call(self.inner)
                        if call:
                            yield "tool", call
                    self.mode, self.inner = None, ""
                    continue
                keep = len(close) - 1
                if len(self.buf) > keep:
                    self.inner += self.buf[:-keep]
                    self.buf = self.buf[-keep:]
                return

    def finish(self) -> Iterator[Tuple[str, object]]:
        if self.mode is None and self.buf:
            yield "text", self.buf
        elif self.mode in ("<tool_call>", "<function="):
            call = parse_tool_call(self.inner + self.buf)
            if call:
                yield "tool", call
        self.buf, self.mode, self.inner = "", None, ""


_XML_FN = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_tool_call(raw: str) -> Optional[dict]:
    raw = raw.strip()
    m = _XML_FN.search(raw)
    if m:
        args = {}
        for k, v in _XML_PARAM.findall(m.group(2)):
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        return {"name": m.group(1).strip(), "arguments": args}
    try:
        obj = json.loads(raw)
    except Exception:
        mm = re.search(r"\{.*\}", raw, re.S)
        if not mm:
            return None
        try:
            obj = json.loads(mm.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return {"name": str(obj["name"]), "arguments": args if isinstance(args, dict) else {}}
