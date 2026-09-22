"""Model registry, on-demand downloads, sizes.

Only the standard library is required for downloading; ``huggingface_hub`` is
used when installed (resume, auth, mirrors).
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .config import Config, cache_root

log = logging.getLogger("sokhan.models")
Progress = Optional[Callable[[str, float], None]]      # (label, 0..1)


@dataclass
class ModelSpec:
    key: str
    kind: str                       # stt | llm | tts | vad
    repo_id: str
    files: List[str] = field(default_factory=list)   # exact names or glob patterns
    approx_disk_mb: int = 0
    approx_ram_mb: int = 0
    note: str = ""


# Sizes are estimates from parameter counts / published file sizes; run
# ``python -m sokhan.report`` after downloading to see the real numbers.
REGISTRY: Dict[str, ModelSpec] = {
    "vad-silero": ModelSpec("vad-silero", "vad", "snakers4/silero-vad", ["silero_vad.onnx"], 2, 30),
    "stt-rizeh": ModelSpec("stt-rizeh", "stt", "Reza2kn/Shenava-Rizeh-v1.0-sherpa-onnx",
                           ["model.onnx", "tokens.txt", "persian_itn.py"], 130, 350,
                           "32M FastConformer CTC (Apache-2.0)"),
    "stt-koochik": ModelSpec("stt-koochik", "stt", "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx",
                             ["model.onnx", "tokens.txt", "persian_itn.py"], 460, 900,
                             "114M FastConformer CTC (Apache-2.0)"),
    "llm-qwen3.5-4b-q4": ModelSpec("llm-qwen3.5-4b-q4", "llm", "unsloth/Qwen3.5-4B-GGUF",
                                   ["*Q4_K_M.gguf"], 2800, 3300, "Q4_K_M quantized"),
    "llm-qwen3.5-4b-mmproj": ModelSpec("llm-qwen3.5-4b-mmproj", "llm", "unsloth/Qwen3.5-4B-GGUF",
                                       ["*mmproj*F16*.gguf"], 700, 800, "vision projector (optional)"),
    "tts-parsigo-onnx": ModelSpec("tts-parsigo-onnx", "tts", "Nimaone/pocket-tts-farsi-v2-onnx",
                                  ["*"], 480, 700, "pocket-tts-farsi-v2 (CC-BY-NC-4.0) incl. ONNX G2P"),
    "tts-parsigo-v2-tokenizer": ModelSpec("tts-parsigo-v2-tokenizer", "tts", "mehdi-hf/pocket-tts-farsi-v2",
                                          ["model.yaml", "normalize_fa.py", "tokenizer_ph.model"], 1, 5,
                                          "SentencePiece tokenizer + text normalizer needed by tts_onnx.py "
                                          "(model.safetensors is NOT needed for the ONNX path)"),
}

SILERO_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
]
TTS_ENGINE_ZIP = "https://github.com/nimaone/persian_tts/archive/refs/heads/main.zip"


# ------------------------------------------------------------ primitives
def _http_get(url: str, dest: str, progress: Progress = None, label: str = "") -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "sokhan/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if progress and total:
                progress(label or os.path.basename(dest), done / total)
    os.replace(tmp, dest)
    return dest


def hf_list(repo: str) -> List[str]:
    try:
        from huggingface_hub import list_repo_files  # type: ignore
        return list(list_repo_files(repo))
    except ImportError:
        pass
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        data = json.loads(r.read().decode())
    return [s["rfilename"] for s in data.get("siblings", [])]


def hf_download(repo: str, filename: str, dest_dir: str, progress: Progress = None) -> str:
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        p = hf_hub_download(repo, filename, local_dir=dest_dir)
        return p
    except ImportError:
        pass
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    return _http_get(url, dest, progress, f"{repo}/{filename}")


def match_files(files: List[str], pattern: str) -> List[str]:
    """Glob over base names; plain F16 is preferred over BF16 (CPU-friendlier), shorter names first."""
    pat = pattern.lower()
    hits = [f for f in files if fnmatch.fnmatch(os.path.basename(f).lower(), pat) or fnmatch.fnmatch(f.lower(), pat)]
    return sorted(hits, key=lambda f: ("bf16" in f.lower(), len(f), f))


# ------------------------------------------------------------ per-model
def ensure_vad(cfg: Config, progress: Progress = None) -> str:
    if cfg.vad.model_path and os.path.exists(cfg.vad.model_path):
        return cfg.vad.model_path
    dest = os.path.join(cache_root(cfg), "vad", "silero_vad.onnx")
    if os.path.exists(dest):
        return dest
    try:  # pip package bundles the file
        import silero_vad  # type: ignore
        p = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    last: Optional[Exception] = None
    for url in SILERO_URLS:
        try:
            return _http_get(url, dest, progress, "silero_vad.onnx")
        except Exception as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"Could not download Silero VAD ({last}). Place silero_vad.onnx at {dest} "
                       f"or set vad.model_path / vad.backend='energy'.")


def stt_repo(cfg: Config) -> str:
    if cfg.stt.repo_id:
        return cfg.stt.repo_id
    return {"rizeh": REGISTRY["stt-rizeh"].repo_id,
            "koochik": REGISTRY["stt-koochik"].repo_id}.get(cfg.stt.model, cfg.stt.model)


def ensure_stt(cfg: Config, progress: Progress = None) -> str:
    """Returns a directory containing model.onnx + tokens.txt (+ persian_itn.py)."""
    if cfg.stt.model_dir:
        return cfg.stt.model_dir
    if os.path.isdir(cfg.stt.model):
        return cfg.stt.model
    repo = stt_repo(cfg)
    dest = os.path.join(cache_root(cfg), "stt", repo.replace("/", "__"))
    for fn in ("model.onnx", "tokens.txt"):
        hf_download(repo, fn, dest, progress)
    try:
        hf_download(repo, "persian_itn.py", dest, progress)
    except Exception:
        log.info("persian_itn.py not available; using built-in ITN")
    return dest


def ensure_llm(cfg: Config, progress: Progress = None, need_mmproj: bool = False) -> tuple[str, str]:
    """Returns (gguf_path, mmproj_path or '')."""
    l = cfg.llm
    if l.model_path and os.path.exists(l.model_path):
        return l.model_path, (l.mmproj_path if need_mmproj else "")
    pattern = l.filename if l.quantized else l.unquantized_filename
    candidates = [l.repo_id] + [r for r in ("bartowski/Qwen_Qwen3.5-4B-GGUF", "lmstudio-community/Qwen3.5-4B-GGUF")
                                if r != l.repo_id]
    dest_root = os.path.join(cache_root(cfg), "llm")
    last: Optional[Exception] = None
    for repo in candidates:
        try:
            files = hf_list(repo)
            hits = [f for f in match_files(files, pattern) if "mmproj" not in f.lower()]
            if not hits:
                continue
            path = hf_download(repo, hits[0], os.path.join(dest_root, repo.replace("/", "__")), progress)
            mm = ""
            if need_mmproj:
                mm = l.mmproj_path
                if not mm:
                    mm_hits = match_files(files, l.mmproj_filename)
                    if mm_hits:
                        mm = hf_download(repo, mm_hits[0], os.path.join(dest_root, repo.replace("/", "__")), progress)
            return path, mm
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not find/download a GGUF matching {pattern!r} in {candidates} ({last}). "
                       f"Set llm.model_path to a local .gguf file.")


def ensure_tts_engine(cfg: Config, progress: Progress = None) -> str:
    """Returns the root of a persian_tts checkout that contains scripts/ and model/onnx/."""
    if cfg.tts.engine_dir and os.path.isdir(cfg.tts.engine_dir):
        root = cfg.tts.engine_dir
    else:
        base = os.path.join(cache_root(cfg), "tts")
        root = os.path.join(base, "persian_tts-main")
        if not os.path.exists(os.path.join(root, "scripts", "tts_onnx.py")):
            zpath = _http_get(TTS_ENGINE_ZIP, os.path.join(base, "persian_tts.zip"), progress, "persian_tts.zip")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(base)
            os.remove(zpath)
    onnx_dir = os.path.join(root, "model", "onnx")
    if not os.path.exists(os.path.join(onnx_dir, "manifest.json")):
        repo = REGISTRY["tts-parsigo-onnx"].repo_id
        for f in hf_list(repo):
            if f.startswith(".") or f.endswith(".md") or f == ".gitattributes":
                continue
            hf_download(repo, f, onnx_dir, progress)
    # tts_onnx.py also needs model/v2/{tokenizer_ph.model, normalize_fa.py, model.yaml}
    # (the SentencePiece tokenizer + text normalizer). model.safetensors from the same
    # repo is the torch checkpoint used only by the export step, NOT by the ONNX path --
    # skip it, it is large and unnecessary here.
    v2_dir = os.path.join(root, "model", "v2")
    if not os.path.exists(os.path.join(v2_dir, "tokenizer_ph.model")):
        spec = REGISTRY["tts-parsigo-v2-tokenizer"]
        for f in spec.files:
            hf_download(spec.repo_id, f, v2_dir, progress)
    return root


# ------------------------------------------------------------ reporting
def dir_size_mb(path: str) -> float:
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total / 2**20


def installed_report(cfg: Config) -> Dict[str, float]:
    root = cache_root(cfg)
    out: Dict[str, float] = {}
    for sub in ("vad", "stt", "llm", "tts"):
        p = os.path.join(root, sub)
        if os.path.exists(p):
            out[sub] = round(dir_size_mb(p), 1)
    return out
