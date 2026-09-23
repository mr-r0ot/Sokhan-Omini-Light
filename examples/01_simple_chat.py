"""The simplest possible voice conversation with Sokhan.

    python examples/01_simple_chat.py

Just talk. Interrupt it whenever you like. Ctrl+C to quit.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # run from a fresh clone

from sokhan import Config, Omni

# 1. Configure. The defaults are complete: 4-bit models on the CPU, downloaded on first run.
config = Config()
config.llm.model = r"C:\Users\tahag\Downloads\Sokhan-Omini-Light-main\Qwen2B"   # local test model;
                                                                                # delete to use the default
# config.tts.backend = "kitten"        # to change the TTS, uncomment these lines (KittenTTS, English;
# config.tts.model = r"C:\Users\tahag\Downloads\Sokhan-Omini-Light-main\kitten_tts_mini_v0_8.onnx"  # voices.npz beside it)
# config.tts.voice = "Luna"          # Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo


config.prompt.greeting = "سلام! چطور می‌تونم کمکتون کنم؟"

# Speak another language? Pick matching speech models in one line:
# config = Config.for_language("en")

# Optional - let it see you through the webcam (needs a vision model + its mmproj file):
# config.vision.enabled = True

# 2. Create the assistant and listen to what happens.
omni = Omni(config)
omni.on("transcript", lambda text: print("You:", text))
omni.on("response_done", lambda text: print("AI :", text))

# 3. Load the models, open the microphone and speakers, and talk.
print("Loading models...")
omni.start()

# from sokhan.vision import Webcam
# Webcam(omni).start()          # every question now comes with a fresh camera frame

omni.listen()
print("Talk to me! (Ctrl+C to quit)")
omni.run_forever()
