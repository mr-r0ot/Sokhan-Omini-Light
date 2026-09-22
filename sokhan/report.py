"""``python -m sokhan.report`` -- disk footprint of every model + a live
hardware/latency summary.  No GPU or network needed if models are already cached."""
from __future__ import annotations

import argparse
import json
import sys

from . import hardware, models
from .config import Config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Sokhan disk/RAM/CPU report")
    ap.add_argument("--config", help="path to a saved Config json/yaml")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    hw = hardware.detect_hardware()
    plan = hardware.plan_resources(cfg, hw)
    installed = models.installed_report(cfg)

    est_disk = {
        "vad (Silero)": 2,
        f"stt ({cfg.stt.model})": models.REGISTRY[f"stt-{cfg.stt.model}"].approx_disk_mb
                                   if f"stt-{cfg.stt.model}" in models.REGISTRY else 460,
        "llm (Qwen3.5-4B Q4_K_M)": 2800 if cfg.llm.quantized else 8200,
        "tts (pocket-tts-farsi-v2 ONNX)": 480,
    }
    if cfg.vision.enabled:
        est_disk["llm mmproj (vision)"] = 700

    report = {
        "hardware": {
            "os": hw.os, "arch": hw.arch, "physical_cores": hw.physical_cores,
            "logical_cores": hw.logical_cores, "total_ram_mb": hw.total_ram_mb,
            "available_ram_mb": hw.avail_ram_mb, "gpus": [g.__dict__ for g in hw.gpus],
        },
        "resource_plan": plan.__dict__,
        "estimated_disk_mb": est_disk,
        "estimated_disk_total_mb": sum(est_disk.values()),
        "already_downloaded_mb": installed,
        "minimum_requirements": {
            "ram_mb": "~3300 (quantized LLM, rizeh STT) / ~9000 (unquantized LLM)",
            "cpu": "2 physical cores (slow but usable) / 4+ recommended for <1.5s replies",
            "disk_mb": sum(est_disk.values()),
            "gpu": "optional; CPU-only is the supported default",
        },
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"Sokhan report\n{'='*13}")
    print(f"Hardware  : {hw.summary()}")
    print(f"Plan      : llm_threads={plan.llm_threads} stt_threads={plan.stt_threads} "
         f"tts_threads={plan.tts_threads} gpu={plan.use_gpu}")
    print("\nEstimated model disk sizes:")
    for k, v in est_disk.items():
        print(f"  {k:<32} ~{v:>6} MB")
    print(f"  {'TOTAL':<32} ~{sum(est_disk.values()):>6} MB")
    if installed:
        print("\nAlready on disk (cache dir):")
        for k, v in installed.items():
            print(f"  {k:<10} {v:>8.1f} MB")
    print("\nMinimum to run comfortably:")
    print("  RAM : ~3.3 GB with default quantized settings (2 CPU cores minimum, 4+ recommended)")
    print("  Disk: ~3.7 GB for all four models")
    print("  GPU : optional (hardware.use_gpu=True); CPU-only is fully supported")
    if plan.notes:
        print("\nNotes:")
        for n in plan.notes:
            print("  -", n)


if __name__ == "__main__":
    main()
