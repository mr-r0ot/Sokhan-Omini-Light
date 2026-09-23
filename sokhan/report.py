"""``python -m sokhan report`` - hardware, resource plan and model disk usage."""
from __future__ import annotations

import json
from typing import Optional

from . import hardware, models
from .config import Config


def build(cfg: Optional[Config] = None) -> dict:
    cfg = cfg or Config()
    hw = hardware.detect()
    plan = hardware.plan(cfg, hw)
    return {
        "hardware": {"summary": hw.summary(), "physical_cores": hw.physical_cores,
                     "total_ram_mb": hw.total_ram_mb, "available_ram_mb": hw.avail_ram_mb,
                     "gpus": [g.__dict__ for g in hw.gpus]},
        "plan": plan.__dict__,
        "models": {"vad": cfg.vad.backend, "stt": f"{cfg.stt.model} [{cfg.stt.quant}]",
                   "llm": f"{cfg.llm.model} [{cfg.llm.quant}]",
                   "tts": f"{cfg.tts.backend}:{cfg.tts.model or 'default'} [{cfg.tts.quant}]"},
        "cache_mb": models.installed(cfg),
    }


def main(cfg: Optional[Config] = None, as_json: bool = False) -> None:
    r = build(cfg)
    if as_json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    print("Sokhan report\n=============")
    print("Hardware :", r["hardware"]["summary"])
    p = r["plan"]
    print(f"Plan     : llm {p['llm_threads']} threads (prefill {p['llm_batch_threads']}), "
          f"stt {p['stt_threads']}, tts {p['tts_threads']}, gpu={p['use_gpu']}")
    for n in p["notes"]:
        print("           -", n)
    print("Models   :")
    for k, v in r["models"].items():
        print(f"  {k:<4} {v}")
    if r["cache_mb"]:
        print("On disk  :", ", ".join(f"{k} {v:.0f} MB" for k, v in r["cache_mb"].items()))
    print("\nTypical footprint with the defaults (4-bit): ~3.2 GB disk, ~3.5 GB RAM (Qwen3.5-4B),"
          " ~2.2 GB RAM with Qwen3.5-2B. No GPU required.")


if __name__ == "__main__":
    main()
