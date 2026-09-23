"""Language packs.

Everything language-specific lives here, so the engine itself is language
neutral. A pack knows how to make text speakable (numbers -> words), how to
post-process transcripts (spoken numbers -> digits), which trailing words mean
"the user is not done yet", and a couple of filler phrases.

Built in: ``fa`` and ``en`` with hand-written rules, plus a generic pack for
every other language (uses ``num2words`` when installed). Add your own::

    from sokhan.lang import LanguagePack, register_language
    register_language(LanguagePack("de", incomplete={"und", "aber", "weil"}, fillers=["Moment..."]))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUM = re.compile(r"\d[\d,٬]*(?:\.\d+)?")


def to_ascii_digits(text: str) -> str:
    return text.translate(_DIGITS)


@dataclass
class LanguagePack:
    code: str
    incomplete: Set[str] = field(default_factory=set)     # trailing words that mean "not finished"
    complete: Set[str] = field(default_factory=set)       # trailing words that mean "done"
    fillers: List[str] = field(default_factory=list)
    number_to_words: Optional[Callable[[int], str]] = None
    itn: Optional[Callable[[str], str]] = None            # transcript post-processing
    pre_normalize: Optional[Callable[[str], str]] = None  # script-level fixes before anything else
    decimal_word: str = "point"
    percent_word: str = "percent"

    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        """Make text speakable for a TTS that cannot read digits."""
        if self.pre_normalize:
            text = self.pre_normalize(text)
        n2w = self.number_to_words or _num2words_for(self.code)
        if n2w is None:
            return text                                     # leave digits: most TTS read them
        text = to_ascii_digits(text)
        text = re.sub(r"(\d+)\s*[%٪]", lambda m: f"{m.group(1)} {self.percent_word}", text)

        def repl(m: re.Match) -> str:
            s = m.group(0).replace(",", "").replace("٬", "")
            if "." in s:
                a, _, b = s.partition(".")
                if a.isdigit() and b.isdigit():
                    frac = " ".join(n2w(int(d)) for d in b) if b.startswith("0") else n2w(int(b))
                    return f"{n2w(int(a))} {self.decimal_word} {frac}"
            if not s.isdigit():
                return m.group(0)
            if len(s) >= 7 and "," not in m.group(0) or len(s) > 15:     # phone numbers / ids
                return ", ".join(" ".join(n2w(int(d)) for d in s[i:i + 3]) for i in range(0, len(s), 3))
            if len(s) > 1 and s.startswith("0"):
                return " ".join(n2w(int(d)) for d in s)
            return n2w(int(s))
        return _NUM.sub(repl, text)

    def completeness(self, text: str) -> float:
        """0..1: does this transcript look like a finished utterance?"""
        t = text.strip()
        if not t:
            return 0.0
        if t[-1] in "?؟!.。？！":
            return 0.95
        words = t.strip("،,、").split()
        if not words:
            return 0.5
        last = words[-1].lower().strip("\"'«»")
        if last in self.incomplete:
            return 0.15
        if last in self.complete:
            return 0.95
        if len(words) >= 6:
            return 0.75
        return 0.6 if len(words) >= 3 else 0.5


_PACKS: Dict[str, LanguagePack] = {}


def register_language(pack: LanguagePack) -> LanguagePack:
    _PACKS[pack.code] = pack
    return pack


def get_language(code: str) -> LanguagePack:
    code = (code or "en").lower().split("-")[0]
    return _PACKS.get(code) or register_language(LanguagePack(code))


def _num2words_for(code: str) -> Optional[Callable[[int], str]]:
    try:
        from num2words import num2words  # type: ignore
    except Exception:
        return None
    try:
        num2words(1, lang=code)
    except Exception:
        return None
    return lambda n: num2words(n, lang=code)


# --------------------------------------------------------------------------- English
_EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
            "fifteen sixteen seventeen eighteen nineteen").split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_EN_SCALES = ["", "thousand", "million", "billion", "trillion"]


def _en_words(n: int) -> str:
    if n < 0:
        return "minus " + _en_words(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        return _EN_TENS[n // 10] + ("-" + _EN_ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _EN_ONES[n // 100] + " hundred" + (" " + _en_words(n % 100) if n % 100 else "")
    parts, i = [], 0
    while n and i < len(_EN_SCALES):
        g = n % 1000
        if g:
            parts.append(_en_words(g) + (" " + _EN_SCALES[i] if i else ""))
        n //= 1000
        i += 1
    return " ".join(reversed(parts))


register_language(LanguagePack(
    "en",
    incomplete={"and", "but", "or", "so", "because", "the", "a", "an", "to", "of", "in", "on", "with", "for",
                "if", "that", "which", "um", "uh", "like", "my", "your", "is", "are", "was", "about", "then",
                "i", "we", "when", "while", "than", "at", "from"},
    complete={"please", "thanks", "thank", "you", "yes", "no", "okay", "ok", "bye", "hello", "hi", "it",
              "done", "now", "today", "tomorrow", "right", "too"},
    fillers=["Hmm...", "One moment."],
    number_to_words=_en_words))


# --------------------------------------------------------------------------- Persian
_AR2FA = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی"})
_FA_ONES = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده", "یازده", "دوازده",
            "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_FA_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_FA_HUND = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_FA_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]


def normalize_fa(text: str) -> str:
    return text.translate(_AR2FA)


def _fa_below_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(_FA_HUND[n // 100])
        n %= 100
    if n >= 20:
        parts.append(_FA_TENS[n // 10])
        n %= 10
        if n:
            parts.append(_FA_ONES[n])
    elif n or not parts:
        parts.append(_FA_ONES[n])
    return " و ".join(p for p in parts if p)


def fa_words(n: int) -> str:
    if n < 0:
        return "منفی " + fa_words(-n)
    if n == 0:
        return "صفر"
    groups, i = [], 0
    while n and i < len(_FA_SCALES):
        g = n % 1000
        if g:
            w = "" if (i == 1 and g == 1) else _fa_below_1000(g)      # "هزار", not "یک هزار"
            groups.append((w + " " + _FA_SCALES[i]).strip() if i else w)
        n //= 1000
        i += 1
    return " و ".join(reversed(groups))


_FA_W2V = {w: i for i, w in enumerate(_FA_ONES)}
_FA_W2V.update({w: 10 * i for i, w in enumerate(_FA_TENS) if w})
_FA_W2V.update({w: 100 * i for i, w in enumerate(_FA_HUND) if w})
_FA_SC = {"هزار": 10 ** 3, "میلیون": 10 ** 6, "میلیارد": 10 ** 9}


def itn_fa(text: str) -> str:
    """Spoken Persian numbers -> digits. Lone words below 10 stay words ('یک' is often an article)."""
    text = normalize_fa(text)
    toks = text.split()
    out: List[str] = []
    i = 0
    while i < len(toks):
        j, total, cur, n, ok = i, 0, 0, 0, False
        while j < len(toks):
            w = toks[j]
            if w == "و" and n and j + 1 < len(toks) and (toks[j + 1] in _FA_W2V or toks[j + 1] in _FA_SC):
                j += 1
                continue
            if w in _FA_SC:
                total += (cur or 1) * _FA_SC[w]
                cur = 0
            elif w in _FA_W2V and not (w == "صفر" and n):
                cur += _FA_W2V[w]
            else:
                break
            n += 1
            ok = True
            j += 1
        if ok and (total + cur >= 10 or n > 1):
            out.append(str(total + cur))
            i = j
            continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


def _fa_pre(text: str) -> str:
    text = normalize_fa(text)
    return re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: f"{m.group(1)} و {m.group(2)} دقیقه" if m.group(2) != "00" else m.group(1),
                  to_ascii_digits(text))


register_language(LanguagePack(
    "fa",
    incomplete={"و", "که", "اما", "ولی", "با", "برای", "از", "به", "در", "را", "رو", "یا", "تا", "اگر", "اگه",
                "چون", "یه", "یک", "این", "آن", "اون", "همین", "بین", "مثل", "هم", "ام", "اِ", "اممم", "آ",
                "خب", "یعنی", "پس", "تو", "توی", "روی", "بعد", "قبل", "بدون", "درباره", "مثلا", "مثلاً",
                "ببین", "راستش", "بعدش", "اینکه", "وقتی", "کدوم", "کدام", "من", "ما", "شما"},
    complete={"است", "هست", "نیست", "میشه", "می‌شه", "می‌شود", "میشود", "چیه", "کیه", "کجاست", "چطوره",
              "چنده", "لطفا", "لطفاً", "ممنون", "متشکرم", "مرسی", "باشه", "بله", "نه", "آره", "خیر",
              "خداحافظ", "سلام", "دارید", "دارین", "داره", "دارد", "میخوام", "می‌خوام", "می‌خواهم",
              "بفرمایید", "کن", "کنید", "بده", "بدید", "بگو", "بگید", "شد", "شده", "ببخشید", "درسته",
              "خوبه", "خوبی", "دارم", "هستم", "بود", "بودم", "کردم", "میکنم", "می‌کنم", "چی", "کجا",
              "چرا", "چطور", "کی", "چند"},
    fillers=["خب...", "یک لحظه."],
    number_to_words=fa_words, itn=itn_fa, pre_normalize=_fa_pre,
    decimal_word="ممیز", percent_word="درصد"))


# --------------------------------------------------------------------------- default models per language
#: Piper voices shipped for sherpa-onnx (https://huggingface.co/csukuangfj). Any other
#: ``vits-piper-<locale>-<voice>`` repo works too - set ``cfg.tts.model`` to it.
PIPER_VOICES = {
    "en": "csukuangfj/vits-piper-en_US-lessac-medium",
    "de": "csukuangfj/vits-piper-de_DE-thorsten-medium",
    "fr": "csukuangfj/vits-piper-fr_FR-siwis-medium",
    "es": "csukuangfj/vits-piper-es_ES-davefx-medium",
    "ar": "csukuangfj/vits-piper-ar_JO-kareem-medium",
    "tr": "csukuangfj/vits-piper-tr_TR-dfki-medium",
    "fa": "csukuangfj/vits-piper-fa_IR-amir-medium",
}
MULTILINGUAL_STT = "csukuangfj/sherpa-onnx-whisper-small"
