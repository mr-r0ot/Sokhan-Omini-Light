"""Hardware detection and CPU/GPU planning.

House rule: the models must never starve each other or the audio path. The
LLM and the TTS run *at the same time* while a reply is being spoken, and
llama.cpp's worker threads spin-wait, so oversubscribing cores is the fastest
way to make everything slow. The plan below splits physical cores between
them instead of letting both grab all of them.
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger("sokhan.hardware")


@dataclass
class GPUInfo:
    kind: str                # cuda | metal | rocm | directml
    name: str = ""
    vram_mb: Optional[int] = None


@dataclass
class HardwareInfo:
    os: str
    arch: str
    logical_cores: int
    physical_cores: int
    total_ram_mb: Optional[int]
    avail_ram_mb: Optional[int]
    gpus: List[GPUInfo] = field(default_factory=list)
    ort_providers: List[str] = field(default_factory=list)
    llama_gpu: bool = False

    def summary(self) -> str:
        g = ", ".join(f"{x.kind}:{x.name}" for x in self.gpus) or "none"
        return (f"{self.os}/{self.arch} | {self.physical_cores} cores ({self.logical_cores} threads) | "
                f"RAM {self.avail_ram_mb}/{self.total_ram_mb} MB free | GPU {g}")


@dataclass
class ResourcePlan:
    llm_threads: int
    llm_batch_threads: int
    stt_threads: int
    tts_threads: int
    use_gpu: bool = False
    llm_gpu_layers: int = 0
    ort_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    notes: List[str] = field(default_factory=list)


class InsufficientMemory(RuntimeError):
    pass


# --------------------------------------------------------------------------- probes
def ram_mb() -> Tuple[Optional[int], Optional[int]]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return int(vm.total / 2**20), int(vm.available / 2**20)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0])
            return info["MemTotal"] // 1024, info.get("MemAvailable", info["MemFree"]) // 1024
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MS()
            st.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return int(st.ullTotalPhys / 2**20), int(st.ullAvailPhys / 2**20)
        if sys.platform == "darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) // 2**20
            return total, None
    except Exception as e:  # pragma: no cover
        log.debug("RAM probe failed: %s", e)
    return None, None


def physical_cores() -> int:
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    logical = os.cpu_count() or 2
    if platform.machine().lower() in ("arm64", "aarch64"):
        return logical
    return max(1, logical // 2)


def _detect_gpus() -> Tuple[List[GPUInfo], List[str], bool]:
    gpus: List[GPUInfo] = []
    providers: List[str] = []
    try:
        import onnxruntime as ort  # type: ignore
        providers = list(ort.get_available_providers())
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                          timeout=4, stderr=subprocess.DEVNULL).decode()
            for line in out.strip().splitlines():
                name, mem = [x.strip() for x in line.split(",")]
                gpus.append(GPUInfo("cuda", name, int(float(mem))))
        except Exception:
            pass
    if not gpus and "CUDAExecutionProvider" in providers:
        gpus.append(GPUInfo("cuda", "CUDA"))
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        gpus.append(GPUInfo("metal", "Apple GPU"))
    if "ROCMExecutionProvider" in providers:
        gpus.append(GPUInfo("rocm", "ROCm"))
    if "DmlExecutionProvider" in providers and not gpus:
        gpus.append(GPUInfo("directml", "DirectML"))
    llama_gpu = False
    try:
        import llama_cpp  # type: ignore
        llama_gpu = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    return gpus, providers, llama_gpu


def detect() -> HardwareInfo:
    total, avail = ram_mb()
    gpus, providers, llama_gpu = _detect_gpus()
    return HardwareInfo(os=platform.system(), arch=platform.machine(), logical_cores=os.cpu_count() or 1,
                        physical_cores=physical_cores(), total_ram_mb=total, avail_ram_mb=avail,
                        gpus=gpus, ort_providers=providers, llama_gpu=llama_gpu)


# --------------------------------------------------------------------------- plan
def plan(cfg, hw: Optional[HardwareInfo] = None) -> ResourcePlan:
    hw = hw or detect()
    cores = max(1, cfg.hardware.threads or hw.physical_cores)
    tts = cfg.tts.threads or (1 if cores <= 4 else 2)
    llm = cfg.llm.threads or max(1, cores - tts)
    p = ResourcePlan(llm_threads=llm, llm_batch_threads=max(llm, cores),
                     stt_threads=cfg.stt.threads or min(2, cores), tts_threads=tts)
    if not cfg.hardware.use_gpu:
        return p
    if not hw.gpus:
        p.notes.append("GPU requested but none found - running on CPU")
        return p
    g = hw.gpus[0]
    p.notes.append(f"GPU: {g.kind} {g.name}".strip())
    if cfg.llm.backend == "llama_cpp" and not hw.llama_gpu:
        p.notes.append("llama-cpp-python is a CPU build - the LLM stays on CPU "
                       "(reinstall with CMAKE_ARGS=\"-DGGML_CUDA=on\" or \"-DGGML_METAL=on\")")
    else:
        p.use_gpu = True
        p.llm_gpu_layers = cfg.llm.gpu_layers
        if g.vram_mb and g.vram_mb < 3500 and cfg.llm.gpu_layers < 0:
            p.llm_gpu_layers = 16
            p.notes.append("small VRAM: partial LLM offload (16 layers)")
    for prov in ("CUDAExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider"):
        if prov in hw.ort_providers:
            p.ort_providers = [prov, "CPUExecutionProvider"]
            p.use_gpu = True
            break
    if p.use_gpu:                          # the GPU carries the heavy lifting: give the CPU back
        p.llm_threads = max(1, min(p.llm_threads, 4))
    return p


def check_memory(cfg, need_mb: int, what: str) -> None:
    """Fail early with a clear message instead of letting the OS thrash."""
    if not cfg.hardware.memory_guard:
        return
    _, avail = ram_mb()
    if avail is None:
        return
    if avail - need_mb < cfg.hardware.min_free_ram_mb:
        raise InsufficientMemory(
            f"Not enough free RAM for {what}: needs ~{need_mb} MB, {avail} MB free. Close other apps, "
            f"use Config.preset('lowmem'), or set hardware.memory_guard = False to try anyway.")


def lower_priority() -> None:
    try:
        if sys.platform == "win32":
            import psutil  # type: ignore
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(3)
    except Exception:
        pass
