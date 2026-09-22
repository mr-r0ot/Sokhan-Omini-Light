"""Persian text utilities: normalisation for TTS, number <-> words, streaming
sentence chunker, tool-call/think stream filter, end-of-turn completeness."""
from __future__ import annotations

import json
import re
from typing import Iterator, List, Optional, Tuple

# ------------------------------------------------------------------ chars
_AR2FA = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی"})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_ZW = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
_MD = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s*|^\s*[-*•]\s+|^\s*\d+\.\s+)", re.M)
_URL = re.compile(r"https?://\S+|www\.\S+")
_PAREN = re.compile(r"\([^)]{0,80}\)|\[[^\]]{0,80}\]")


def normalize_chars(text: str) -> str:
    return text.translate(_AR2FA)


def to_ascii_digits(text: str) -> str:
    return text.translate(_DIGITS)


# ------------------------------------------------------------------ numbers -> words
_ONES = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده", "یازده", "دوازده",
         "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUND = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]


def _below_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(_HUND[n // 100]); n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10]); n %= 10
        if n:
            parts.append(_ONES[n])
    elif n > 0 or not parts:
        if n or not parts:
            parts.append(_ONES[n])
    return " و ".join(p for p in parts if p)


def int_to_words(n: int) -> str:
    if n < 0:
        return "منفی " + int_to_words(-n)
    if n == 0:
        return "صفر"
    groups, i = [], 0
    while n and i < len(_SCALES):
        g = n % 1000
        if g:
            w = _below_1000(g)
            if i == 1 and g == 1:
                w = ""                      # "هزار و ..." not "یک هزار و ..."
            groups.append((w + " " + _SCALES[i]).strip() if i else w)
        n //= 1000; i += 1
    return " و ".join(reversed(groups))


def _num_repl(m: re.Match) -> str:
    raw = m.group(0)
    s = raw.replace(",", "").replace("٬", "").replace("،", "")
    if "." in s or "/" in s:
        sep = "." if "." in s else "/"
        a, _, b = s.partition(sep)
        if a.isdigit() and b.isdigit() and len(b) <= 3:
            return f"{int_to_words(int(a))} ممیز {' '.join(_ONES[int(d)] for d in b) if b.startswith('0') else int_to_words(int(b))}"
    if not s.isdigit():
        return raw
    if len(s) >= 7 and "," not in raw:          # phone / long ids: digit by digit, grouped
        return "، ".join(" ".join(_ONES[int(d)] for d in s[i:i + 3]) for i in range(0, len(s), 3))
    if len(s) > 1 and s.startswith("0"):
        return " ".join(_ONES[int(d)] for d in s)
    if len(s) > 15:
        return " ".join(_ONES[int(d)] for d in s)
    return int_to_words(int(s))


_NUM_RE = re.compile(r"\d[\d,٬،]*(?:[./]\d+)?")


def expand_numbers(text: str) -> str:
    text = to_ascii_digits(text)
    text = re.sub(r"(\d+)\s*[%٪]", lambda m: m.group(1) + " درصد", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{m.group(1)} و {m.group(2)} دقیقه" if m.group(2) != "00" else f"{m.group(1)}", text)
    return _NUM_RE.sub(_num_repl, text)


def clean_for_tts(text: str) -> str:
    """Make LLM output speakable: strip markdown/emoji/URLs/brackets, spell numbers."""
    t = _ZW.sub(" ", text)
    t = _URL.sub(" ", t)
    t = _MD.sub("", t)
    t = _PAREN.sub(" ", t)
    t = _EMOJI.sub("", t)
    t = normalize_chars(t)
    t = expand_numbers(t)
    t = t.replace("&", " و ").replace("@", " ات ").replace("_", " ").replace("*", " ").replace("#", " ")
    t = re.sub(r"[\"“”«»<>{}|~^=+\\/]", " ", t)
    t = re.sub(r"\s*\n+\s*", ". ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"([.!?؟،,;؛:])\1+", r"\1", t)
    return t


def has_speakable(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FFA-Za-z0-9]", text))


# ------------------------------------------------------------------ words -> digits (ITN fallback)
_W2V = {w: i for i, w in enumerate(_ONES)}
_W2V.update({w: 10 * i for i, w in enumerate(_TENS) if w})
_W2V.update({w: 100 * i for i, w in enumerate(_HUND) if w})
_SC = {"هزار": 10 ** 3, "میلیون": 10 ** 6, "میلیارد": 10 ** 9}


def itn_fa(text: str) -> str:
    """Spoken Persian numbers -> digits.  Single words below 10 stay as words
    (safer: 'یک' is often the indefinite article)."""
    toks = text.split()
    out: List[str] = []
    i = 0
    while i < len(toks):
        j, total, cur, n_tok, ok = i, 0, 0, 0, False
        while j < len(toks):
            w = toks[j]
            if w == "و" and n_tok and j + 1 < len(toks) and (toks[j + 1] in _W2V or toks[j + 1] in _SC):
                j += 1
                continue
            if w in _SC:
                total += (cur or 1) * _SC[w]; cur = 0
            elif w in _W2V and not (w == "صفر" and n_tok):
                cur += _W2V[w]
            else:
                break
            n_tok += 1; ok = True; j += 1
        if ok:
            val = total + cur
            if val >= 10 or n_tok > 1:
                out.append(str(val)); i = j; continue
        out.append(toks[i]); i += 1
    return " ".join(out)


# ------------------------------------------------------------------ chunking for TTS
_STRONG = "\n.!?؟…"
_WEAK = "،,;؛:"


class SentenceChunker:
    """Turns a token stream into speakable chunks with minimal first-audio latency.

    * first chunk: as soon as a clause boundary appears after ``first_min`` chars
    * later chunks: sentence ends, or a clause boundary once ``min_chars`` reached
    * hard cap ``max_chars`` (split at the last space)
    """

    def __init__(self, first_min: int = 14, min_chars: int = 30, max_chars: int = 150):
        self.first_min, self.min_chars, self.max_chars = first_min, min_chars, max_chars
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
            cut = self._find_cut(final)
            if cut is None:
                break
            chunk, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if chunk and has_speakable(chunk):
                out.append(chunk); self.emitted += 1
        if final and self.buf.strip():
            rest = self.buf.strip()
            self.buf = ""
            if has_speakable(rest):
                out.append(rest); self.emitted += 1
        return out

    def _find_cut(self, final: bool) -> Optional[int]:
        b = self.buf
        need = self.first_min if self.emitted == 0 else self.min_chars
        for i, ch in enumerate(b):
            if ch in _STRONG:
                if ch == "." and 0 < i < len(b) - 1 and b[i - 1].isdigit() and b[i + 1].isdigit():
                    continue                                    # decimal point
                if i + 1 >= len(b) and not final and ch in ".":  # wait: might be part of "..." or "3."
                    return None
                if i + 1 >= need or ch in "\n":
                    return i + 1
            elif ch in _WEAK and i + 1 >= need and i + 1 < len(b):
                return i + 1
        if len(b) > self.max_chars:
            sp = b.rfind(" ", 0, self.max_chars)
            return (sp if sp > 20 else self.max_chars)
        return None


# ------------------------------------------------------------------ LLM stream filter
_TAGS = ("<think>", "<tool_call>")
_CLOSE = {"<think>": "</think>", "<tool_call>": "</tool_call>"}


class StreamFilter:
    """Splits a raw LLM stream into speakable text and structured tool calls.

    ``<think>...</think>`` is dropped; ``<tool_call>{json}</tool_call>`` is parsed.
    Partial tags at chunk boundaries are held back correctly.
    """

    def __init__(self, start_in_think: bool = False):
        self.buf = ""
        self.mode: Optional[str] = "<think>" if start_in_think else None   # None | "<think>" | "<tool_call>"
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
                hit = next((t for t in _TAGS if self.buf.startswith(t)), None)
                if hit:
                    self.mode, self.inner = hit, ""
                    self.buf = self.buf[len(hit):]
                    continue
                if any(t.startswith(self.buf) for t in _TAGS):
                    return                                  # could still become a tag: wait
                yield "text", self.buf[0]
                self.buf = self.buf[1:]
            else:
                close = _CLOSE[self.mode]
                pos = self.buf.find(close)
                if pos >= 0:
                    self.inner += self.buf[:pos]
                    self.buf = self.buf[pos + len(close):]
                    if self.mode == "<tool_call>":
                        call = _parse_call(self.inner)
                        if call:
                            yield "tool", call
                    self.mode, self.inner = None, ""
                    continue
                keep = len(close) - 1                       # keep a possible partial closing tag
                if len(self.buf) > keep:
                    self.inner += self.buf[:-keep]
                    self.buf = self.buf[-keep:]
                return

    def finish(self) -> Iterator[Tuple[str, object]]:
        if self.mode is None and self.buf:
            yield "text", self.buf
        elif self.mode == "<tool_call>":
            call = _parse_call(self.inner + self.buf)
            if call:
                yield "tool", call
        self.buf, self.mode, self.inner = "", None, ""


def _parse_call(raw: str) -> Optional[dict]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
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
    return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}


# ------------------------------------------------------------------ turn completeness
_INCOMPLETE_END = {"و", "که", "اما", "ولی", "با", "برای", "از", "به", "در", "را", "رو", "یا", "تا", "اگر", "اگه",
                   "چون", "یه", "یک", "این", "آن", "اون", "همین", "بین", "مثل", "هم", "ام", "اِ", "اممم", "آ", "خب",
                   "یعنی", "پس", "تو", "توی", "روی", "بعد", "قبل", "چه", "چی", "کدوم", "کدام", "ازش", "بهش", "من", "ما",
                   "شما", "او", "اینکه", "چرا", "کجا", "کی", "چطور", "چگونه", "چند"}
_COMPLETE_END = {"است", "هست", "نیست", "میشه", "می‌شه", "می‌شود", "میشود", "چیه", "کیه", "کجاست", "چطوره", "چنده",
                 "لطفا", "لطفاً", "ممنون", "متشکرم", "مرسی", "باشه", "بله", "نه", "آره", "خیر", "خداحافظ", "سلام",
                 "دارید", "دارین", "داره", "دارد", "میخوام", "می‌خوام", "می‌خواهم", "میخواهم", "بفرمایید", "کن", "کنید",
                 "بده", "بدید", "بگو", "بگید", "شد", "شده", "ببخشید", "درسته", "اوکی", "خوبه", "خوبی", "دارم", "هستم",
                 "بود", "بودم", "کردم", "میکنم", "می‌کنم"}


def completeness(text: str) -> float:
    """0..1 heuristic: does this transcript look like a finished utterance?"""
    t = text.strip().strip("؟?!.،,")
    if not t:
        return 0.0
    words = t.split()
    last = words[-1]
    if last in _INCOMPLETE_END:
        return 0.15
    if last in _COMPLETE_END:
        return 0.95
    if len(words) >= 6:
        return 0.75
    if len(words) >= 3:
        return 0.6
    return 0.45
