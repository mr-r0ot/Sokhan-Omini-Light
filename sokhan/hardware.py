"""Hardware detection, GPU auto-detect (opt-in) and CPU/RAM planning.

The rule of the house: *never* let the models starve the audio path.  We keep
a few cores free for capture / VAD / playback and cap every component's thread
count so the OS stays responsive.
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
from typing import List, Optional

log = logging.getLogger("sokhan.hw")


@dataclass
class GPUInfo:
    kind: str                # "cuda" | "metal" | "directml" | "rocm" | "none"
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
    llama_gpu_offload: bool = False

    def summary(self) -> str:
        g = ", ".join(f"{x.kind}:{x.name}" for x in self.gpus) or "none"
        return (f"{self.os}/{self.arch} cores={self.physical_cores}p/{self.logical_cores}l "
                f"ram={self.avail_ram_mb}/{self.total_ram_mb}MB gpu={g}")


@dataclass
class ResourcePlan:
    llm_threads: int
    llm_batch_threads: int
    stt_threads: int
    tts_threads: int
    llm_gpu_layers: int          # 0 = CPU
    stt_provider: str            # "cpu" | "cuda" | ...
    use_gpu: bool
    notes: List[str] = field(default_factory=list)


class InsufficientMemory(RuntimeError):
    pass


# ---------------------------------------------------------------- RAM
def _ram_mb() -> tuple[Optional[int], Optional[int]]:
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
                    info[k] = int(v.strip().split()[0])
            return info["MemTotal"] // 1024, info.get("MemAvailable", info["MemFree"]) // 1024
        if sys.platform == "win32":
            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return int(st.ullTotalPhys / 2**20), int(st.ullAvailPhys / 2**20)
        if sys.platform == "darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) // 2**20
            return total, None
    except Exception as e:  # pragma: no cover
        log.debug("ram probe failed: %s", e)
    return None, None


def _physical_cores() -> int:
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    logical = os.cpu_count() or 2
    # SMT is the norm on x86; Apple/ARM are usually 1 thread per core.
    if platform.machine().lower() in ("arm64", "aarch64"):
        return logical
    return max(1, logical // 2)


# ---------------------------------------------------------------- GPU
def _detect_gpus() -> tuple[List[GPUInfo], List[str], bool]:
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
            out = subprocess.check_output(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                timeout=4, stderr=subprocess.DEVNULL).decode()
            for line in out.strip().splitlines():
                name, mem = [x.strip() for x in line.split(",")]
                gpus.append(GPUInfo("cuda", name, int(float(mem))))
        except Exception:
            pass
    if not gpus and "CUDAExecutionProvider" in providers:
        gpus.append(GPUInfo("cuda", "cuda-provider"))
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        gpus.append(GPUInfo("metal", "Apple GPU"))
    if "DmlExecutionProvider" in providers and not any(g.kind == "cuda" for g in gpus):
        gpus.append(GPUInfo("directml", "DirectML"))
    if "ROCMExecutionProvider" in providers:
        gpus.append(GPUInfo("rocm", "ROCm"))
    llama_off = False
    try:
        import llama_cpp  # type: ignore
        llama_off = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    return gpus, providers, llama_off


def detect_hardware() -> HardwareInfo:
    total, avail = _ram_mb()
    gpus, providers, llama_off = _detect_gpus()
    return HardwareInfo(
        os=platform.system(), arch=platform.machine(),
        logical_cores=os.cpu_count() or 1, physical_cores=_physical_cores(),
        total_ram_mb=total, avail_ram_mb=avail, gpus=gpus,
        ort_providers=providers, llama_gpu_offload=llama_off)


# ---------------------------------------------------------------- plan
def plan_resources(cfg, hw: Optional[HardwareInfo] = None) -> ResourcePlan:
    hw = hw or detect_hardware()
    h, notes = cfg.hardware, []
    cores = h.cpu_threads or hw.physical_cores
    cores = max(1, cores)

    # Budget: 1 core for audio+VAD+UI, 2 for TTS ONNX (its own intra-op setting),
    # 1-3 for STT bursts.  The LLM gets the rest.  STT and LLM rarely overlap
    # (STT runs while the LLM is idle) so they may share.
    tts_threads = 2 if cores >= 4 else 1
    stt_threads = cfg.stt.num_threads or max(1, min(4, cores // 2))
    llm_threads = cfg.llm.n_threads or max(1, cores - 1 - (1 if cores >= 6 else 0))
    if cores >= 8 and not cfg.llm.n_threads:
        llm_threads = cores - 2      # keep two cores free for TTS/audio
    llm_batch_threads = max(llm_threads, min(cores, llm_threads + 1))

    use_gpu, gpu_layers, stt_provider = False, 0, "cpu"
    if h.use_gpu:
        if hw.gpus:
            g = hw.gpus[0]
            notes.append(f"GPU enabled: {g.kind} {g.name}")
            if hw.llama_gpu_offload or cfg.llm.backend != "llama_cpp":
                gpu_layers, use_gpu = -1, True
            else:
                notes.append("llama-cpp-python was built without GPU support; LLM stays on CPU "
                             "(install a GPU build or use backend='llama_server').")
            if g.kind == "cuda" and "CUDAExecutionProvider" in hw.ort_providers:
                stt_provider, use_gpu = "cuda", True
            if g.kind == "metal" and cfg.llm.backend == "llama_cpp" and hw.llama_gpu_offload:
                gpu_layers, use_gpu = -1, True
            if g.vram_mb and g.vram_mb < 3500 and gpu_layers:
                gpu_layers = 20
                notes.append("small VRAM: partial offload (20 layers)")
        else:
            notes.append("GPU requested but none detected -> running on CPU")
    return ResourcePlan(llm_threads, llm_batch_threads, stt_threads, tts_threads,
                        gpu_layers, stt_provider, use_gpu, notes)


# ---------------------------------------------------------------- guard
def check_memory(cfg, need_mb: int, what: str, hw: Optional[HardwareInfo] = None) -> None:
    """Fail early (with a clear message) instead of letting the OS thrash/kill us."""
    if not cfg.hardware.memory_guard:
        return
    _, avail = _ram_mb()
    if avail is None:
        return
    margin = cfg.hardware.min_free_ram_mb
    if avail - need_mb < margin:
        raise InsufficientMemory(
            f"Not enough free RAM to load {what}: need ~{need_mb} MB, only {avail} MB free "
            f"(keeping {margin} MB spare). Close other apps, use the 'lowmem' preset, "
            f"or set hardware.memory_guard=False.")


def lower_priority() -> None:
    """Nice the *current thread's process* slightly so the UI/audio stays fluid."""
    try:
        if sys.platform == "win32":
            import psutil  # type: ignore
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(3)
    except Exception:
        pass
