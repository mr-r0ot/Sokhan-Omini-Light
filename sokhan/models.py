"""Model resolution: local paths, Hugging Face downloads, and one-time quantization.

Everything is cached under ``~/.cache/sokhan`` (``$SOKHAN_HOME`` or
``config.cache_dir`` to move it). Downloads resume after interruptions, honour
``HF_TOKEN`` / ``HF_ENDPOINT`` (mirrors), and report progress through the
engine's ``load_progress`` event. After the first run everything is offline.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config, cache_root

log = logging.getLogger("sokhan.models")
Progress = Optional[Callable[[str, float, str], None]]          # (stage, 0..1, message)

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SILERO_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
]
# Pocket-TTS Farsi pure-ONNX engine, pinned to a known-good commit.
POCKET_ENGINE_SHA = "e834bd65ac622f376c4d2050821decf5eb95241c"
POCKET_ENGINE_ZIP = "https://github.com/nimaone/persian_tts/archive/{ref}.zip"
POCKET_ONNX_REPO = "Nimaone/pocket-tts-farsi-v2-onnx"
POCKET_TOKENIZER_REPO = "mehdi-hf/pocket-tts-farsi-v2"
POCKET_TOKENIZER_FILES = ["model.yaml", "normalize_fa.py", "tokenizer_ph.model"]
DEFAULT_SHERPA_TTS = "csukuangfj/vits-piper-fa_IR-amir-medium"

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _headers() -> Dict[str, str]:
    h = {"User-Agent": "sokhan/1.0"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


# --------------------------------------------------------------------------- HTTP
def download(url: str, dest: str, progress: Progress = None, stage: str = "download",
             retries: int = 4) -> str:
    """Resumable download to ``dest`` (atomic rename on completion)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    name = os.path.basename(dest)
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = dict(_headers(), **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if have and r.status != 206:           # server ignored the range: start over
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                done, last = have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        now = time.monotonic()
                        if progress and total and now - last > 0.25:
                            last = now
                            progress(stage, done / total, f"downloading {name} "
                                     f"({done / 2**20:.0f}/{total / 2**20:.0f} MB)")
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:                 # already complete
                os.replace(part, dest)
                return dest
            if e.code in (401, 403, 404):
                raise RuntimeError(f"download failed ({e.code}) for {url}") from e
            err: Exception = e
        except Exception as e:                         # network hiccup: back off and resume
            err = e
        log.warning("download of %s failed (%s), retrying...", name, err)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {err}")


def hf_files(repo: str) -> List[str]:
    url = f"{HF}/api/models/{repo}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=30) as r:
            data = json.loads(r.read().decode())
        return [s["rfilename"] for s in data.get("siblings", [])]
    except Exception as e:
        try:
            from huggingface_hub import list_repo_files  # type: ignore
            return list(list_repo_files(repo))
        except Exception:
            raise RuntimeError(f"cannot list Hugging Face repo {repo!r}: {e}") from e


def hf_file(repo: str, filename: str, dest_dir: str, progress: Progress = None, stage: str = "") -> str:
    dest = os.path.join(dest_dir, *filename.split("/"))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    with _lock(dest):
        try:
            return download(f"{HF}/{repo}/resolve/main/{filename}", dest, progress, stage or repo)
        except RuntimeError:
            from huggingface_hub import hf_hub_download  # type: ignore  (auth, mirrors, xet)
            return hf_hub_download(repo, filename, local_dir=dest_dir)


def repo_dir(cfg: Config, kind: str, repo: str) -> str:
    return os.path.join(cache_root(cfg), kind, repo.replace("/", "__"))


def is_local(ref: str) -> bool:
    return bool(ref) and (os.path.exists(ref) or ref.startswith((".", "/", "~")) or ":\\" in ref or ":/" in ref)


# --------------------------------------------------------------------------- quantization
def quantize_q4(src: str, dst: str, progress: Progress = None, stage: str = "") -> Optional[str]:
    """4-bit block quantization of every MatMul weight (onnxruntime MatMulNBits).

    Done once and cached next to the source. Returns None (-> use fp32) when the
    optional ``onnx`` package is missing or the graph cannot be quantized.
    """
    if os.path.exists(dst):
        return dst
    with _lock(dst):
        if os.path.exists(dst):
            return dst
        try:
            import onnx  # type: ignore
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer  # type: ignore
        except Exception:
            log.info("`onnx` not installed: using %s unquantized (pip install onnx for 4-bit)",
                     os.path.basename(src))
            return None
        if progress:
            progress(stage, 0.5, f"quantizing {os.path.basename(src)} to 4-bit (one time)")
        qlog = logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer")
        old = qlog.level
        qlog.setLevel(logging.WARNING)
        try:
            model = onnx.load(src)
            q = MatMulNBitsQuantizer(model, block_size=32, is_symmetric=True, accuracy_level=4)
            q.process()
            tmp = dst + ".tmp"
            q.model.save_model_to_file(tmp, use_external_data_format=False)
            os.replace(tmp, dst)
            return dst
        except Exception as e:
            log.warning("4-bit quantization of %s failed (%s); using fp32", os.path.basename(src), e)
            return None
        finally:
            qlog.setLevel(old)


def pick_quant(src: str, quant: str, progress: Progress = None, stage: str = "") -> str:
    """Return the model file to load for ``quant`` ("q4" | "int8" | "fp32")."""
    quant = (quant or "fp32").lower()
    base, ext = os.path.splitext(src)
    if quant in ("int8", "q8"):
        for cand in (base + ".int8" + ext, base + "_int8" + ext):
            if os.path.exists(cand):
                return cand
        quant = "q4"
    if quant in ("q4", "int4", "4bit"):
        return quantize_q4(src, base + ".q4" + ext, progress, stage) or src
    return src


# --------------------------------------------------------------------------- VAD
def resolve_vad(cfg: Config, progress: Progress = None) -> str:
    if cfg.vad.model:
        return cfg.vad.model
    dest = os.path.join(cache_root(cfg), "vad", "silero_vad.onnx")
    if os.path.exists(dest):
        return dest
    try:                                              # the pip package bundles the file
        import silero_vad  # type: ignore
        p = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    last: Optional[Exception] = None
    for url in SILERO_URLS:
        try:
            return download(url, dest, progress, "vad")
        except Exception as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"could not download Silero VAD ({last}); set vad.backend = 'energy'")


# --------------------------------------------------------------------------- STT
_STT_SKIP = re.compile(r"(\.md$|^\.git|/test_wavs/|\.wav$|\.mp3$|\.flac$|\.png$|\.jpg$|^export|\.sh$)")


def resolve_stt(cfg: Config, progress: Progress = None) -> str:
    """Directory holding the recognizer files."""
    ref = cfg.stt.model
    if is_local(ref):
        if not os.path.isdir(ref):
            raise FileNotFoundError(f"stt.model directory not found: {ref}")
        return ref
    dest = repo_dir(cfg, "stt", ref)
    marker = os.path.join(dest, ".complete")
    if os.path.exists(marker) or (os.path.exists(os.path.join(dest, "tokens.txt"))
                                  and any(f.endswith(".onnx") for f in os.listdir(dest))):
        return dest                                    # cached: no network needed
    files = [f for f in hf_files(ref) if not _STT_SKIP.search(f)]
    onnx_files = [f for f in files if f.endswith(".onnx")]
    want_int8 = cfg.stt.quant.lower() in ("int8", "q8")
    has_int8 = any(".int8." in f for f in onnx_files)
    keep = []
    for f in files:
        if f.endswith(".onnx") and has_int8 and len(onnx_files) > 1:
            if want_int8 != (".int8." in f):
                continue
        keep.append(f)
    for i, f in enumerate(keep):
        hf_file(ref, f, dest, progress, "stt")
        if progress:
            progress("stt", (i + 1) / len(keep), f"stt: {f}")
    open(marker, "w").close()
    return dest


# --------------------------------------------------------------------------- LLM
def _gguf_rank(fname: str, quant: str) -> Tuple[int, int]:
    low = os.path.basename(fname).lower()
    q = quant.lower()
    exact = low.endswith(f"-{q}.gguf") or low.endswith(f"_{q}.gguf") or f"-{q}-00001-of" in low
    return (0 if exact else 1, len(low))


def _cached_gguf(dest: str, quant: str, need_mmproj: bool) -> Optional[Tuple[str, str]]:
    if not os.path.isdir(dest):
        return None
    names = [f for f in os.listdir(dest) if f.lower().endswith(".gguf")]
    main = sorted((f for f in names if "mmproj" not in f.lower() and quant.lower() in f.lower()),
                  key=lambda f: _gguf_rank(f, quant))
    if not main:
        return None
    if re.search(r"-(\d{5})-of-(\d{5})\.gguf$", main[0]):
        return None                                   # shards: let the full resolver verify them
    mm = sorted(f for f in names if "mmproj" in f.lower())
    if need_mmproj and not mm:
        return None
    return os.path.join(dest, main[0]), (os.path.join(dest, mm[0]) if need_mmproj else "")


def resolve_llm(cfg: Config, progress: Progress = None, need_mmproj: bool = False) -> Tuple[str, str]:
    """-> (gguf path, mmproj path or "")."""
    l = cfg.llm
    if is_local(l.model):
        if not os.path.isfile(l.model):
            raise FileNotFoundError(f"llm.model file not found: {l.model}")
        mm = l.mmproj
        if need_mmproj and not mm:
            folder = os.path.dirname(os.path.abspath(l.model))
            cands = sorted(f for f in os.listdir(folder) if "mmproj" in f.lower() and f.endswith(".gguf"))
            mm = os.path.join(folder, cands[0]) if cands else ""
        return l.model, mm
    quant = l.quant or "Q4_K_M"
    dest = repo_dir(cfg, "llm", l.model)
    cached = _cached_gguf(dest, quant, need_mmproj)       # offline after the first download
    if cached:
        return cached
    files = hf_files(l.model)
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    hits = [f for f in ggufs if quant.lower() in os.path.basename(f).lower()]
    if not hits:
        raise RuntimeError(f"no {quant} GGUF in {l.model}. Available: "
                           f"{', '.join(os.path.basename(f) for f in ggufs[:12])}")
    hits.sort(key=lambda f: _gguf_rank(f, quant))
    first = hits[0]
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    parts = [first]
    if m:                                             # split GGUF: fetch every shard
        stem = first[: m.start()]
        parts = sorted(f for f in ggufs if f.startswith(stem) and re.search(r"-\d{5}-of-\d{5}\.gguf$", f))
    path = ""
    for p in parts:
        got = hf_file(l.model, p, dest, progress, "llm")
        path = path or got
    mm = l.mmproj
    if need_mmproj and not mm:
        mms = [f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")]
        mms.sort(key=lambda f: (0 if "f16" in f.lower() and "bf16" not in f.lower() else
                                1 if "bf16" in f.lower() else 2, len(f)))
        if mms:
            mm = hf_file(l.model, mms[0], dest, progress, "llm")
    return path, mm


# --------------------------------------------------------------------------- TTS
def resolve_pocket_tts(cfg: Config, progress: Progress = None) -> str:
    """Root of the Pocket-TTS Farsi ONNX engine (scripts/ + model/onnx + model/v2)."""
    ref = cfg.tts.model
    if is_local(ref):
        root = ref
    else:
        base = os.path.join(cache_root(cfg), "tts")
        root = ""
        for cand in (f"persian_tts-{POCKET_ENGINE_SHA}", "persian_tts-main"):
            if os.path.exists(os.path.join(base, cand, "scripts", "tts_onnx.py")):
                root = os.path.join(base, cand)
                break
        if not root:
            with _lock("pocket-engine"):
                z = download(POCKET_ENGINE_ZIP.format(ref=POCKET_ENGINE_SHA),
                             os.path.join(base, "persian_tts.zip"), progress, "tts")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(base)
                os.remove(z)
                root = os.path.join(base, f"persian_tts-{POCKET_ENGINE_SHA}")
    onnx_dir = os.path.join(root, "model", "onnx")
    if not os.path.exists(os.path.join(onnx_dir, "manifest.json")):
        files = [f for f in hf_files(POCKET_ONNX_REPO) if not f.startswith(".") and not f.endswith(".md")]
        for i, f in enumerate(files):
            hf_file(POCKET_ONNX_REPO, f, onnx_dir, progress, "tts")
            if progress:
                progress("tts", (i + 1) / len(files), f"tts: {f}")
    v2 = os.path.join(root, "model", "v2")
    if not os.path.exists(os.path.join(v2, "tokenizer_ph.model")):
        for f in POCKET_TOKENIZER_FILES:          # the ONNX path needs these, not the torch checkpoint
            hf_file(POCKET_TOKENIZER_REPO, f, v2, progress, "tts")
    return root


def resolve_snapshot(cfg: Config, kind: str, ref: str, progress: Progress = None) -> str:
    """Whole-repo download (sherpa-onnx TTS models ship espeak-ng data etc.)."""
    if is_local(ref):
        return ref
    dest = repo_dir(cfg, kind, ref)
    marker = os.path.join(dest, ".complete")
    if os.path.exists(marker):
        return dest
    files = [f for f in hf_files(ref) if not f.startswith(".git") and not f.endswith((".md", ".wav"))]
    from concurrent.futures import ThreadPoolExecutor
    done = [0]

    def get(f: str) -> None:                     # many small files (espeak data): fetch in parallel
        hf_file(ref, f, dest, None, kind)
        done[0] += 1
        if progress and (done[0] % 10 == 0 or done[0] == len(files)):
            progress(kind, done[0] / len(files), f"{kind}: {done[0]}/{len(files)} files")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(get, files))
    open(marker, "w").close()
    return dest


# --------------------------------------------------------------------------- report
def dir_size_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total / 2**20


def installed(cfg: Config) -> Dict[str, float]:
    root = cache_root(cfg)
    return {sub: round(dir_size_mb(os.path.join(root, sub)), 1)
            for sub in ("vad", "stt", "llm", "tts") if os.path.exists(os.path.join(root, sub))}


def clear_cache(cfg: Config, kind: str) -> None:
    shutil.rmtree(os.path.join(cache_root(cfg), kind), ignore_errors=True)


def match(files: List[str], pattern: str) -> List[str]:
    return [f for f in files if fnmatch.fnmatch(os.path.basename(f).lower(), pattern.lower())]
