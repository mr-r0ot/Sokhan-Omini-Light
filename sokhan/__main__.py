"""Command line.

    python -m sokhan                      talk through the microphone (default config)
    python -m sokhan chat                 text chat in the terminal
    python -m sokhan report               hardware, plan, model sizes
    python -m sokhan --config my.json --llm-model path/to/model.gguf --language en
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import Config, Omni, __version__


def _utf8_console() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None) -> None:
    _utf8_console()
    ap = argparse.ArgumentParser(prog="sokhan", description=f"Sokhan {__version__} realtime voice assistant")
    ap.add_argument("mode", nargs="?", default="voice", choices=["voice", "chat", "report"])
    ap.add_argument("--config", help="JSON/YAML config file")
    ap.add_argument("--language", help="ISO code, e.g. fa, en, de (picks matching speech models)")
    ap.add_argument("--llm-model", help="GGUF file or Hugging Face repo")
    ap.add_argument("--voice", help="built-in voice or a ~5 s WAV to clone")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s: %(message)s")

    cfg = Config.load(a.config) if a.config else Config()
    if a.language:
        cfg = Config.for_language(a.language, cfg)
    if a.llm_model:
        cfg.llm.model = a.llm_model
    if a.voice:
        cfg.tts.voice = a.voice
    cfg.hardware.use_gpu = cfg.hardware.use_gpu or a.gpu
    if a.mode == "report":
        from .report import main as report
        report(cfg)
        return

    omni = Omni(cfg)
    omni.on("load_progress", lambda stage, fraction, message: print(f"\r  {message:<70}", end="", flush=True))
    print("Loading models (the first run downloads them)...")
    omni.start()
    print("\r" + " " * 74 + "\rReady.", omni.capabilities)
    omni.on("transcript", lambda text: print(f"\nyou > {text}"))
    omni.on("response_delta", lambda text: print(text, end="", flush=True))
    omni.on("response_done", lambda text, interrupted: print(" [interrupted]" if interrupted else ""))
    omni.on("metrics", lambda latency_ms: print(f"      ({latency_ms:.0f} ms)"))
    if a.mode == "chat":
        try:
            while True:
                text = input("\nyou > ").strip()
                if text in ("exit", "quit"):
                    break
                if text:
                    print("bot > ", end="")
                    omni.ask(text)
        except (EOFError, KeyboardInterrupt):
            pass
        omni.close()
        return
    omni.on("response_start", lambda: print("bot > ", end=""))
    omni.listen()
    print("Listening. Talk any time, interrupt it whenever you like. Ctrl+C to quit.")
    omni.run_forever()


if __name__ == "__main__":
    main()
