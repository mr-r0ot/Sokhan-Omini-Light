"""Omni - the realtime conversation engine.

    mic ─► VAD ─► endpointer ─┬─ pause ─► STT ─► speculative turn ─► LLM ─► chunker ─► TTS front-end ─► TTS ─┐
                              │                    (audio held back until the turn is confirmed)               │
                              ├─ resume ─► drop the speculation silently                                       │
                              └─ end ────► confirm: held audio plays at once ──────────────────► speaker ◄─────┘
               barge-in ◄──── user talks over the assistant: stop, remember what was heard

Why it feels like one omni model rather than a pipeline:

* **Speculative turns.** The first ~240 ms pause in the user's speech is
  transcribed (tens of ms) and the reply is generated and synthesised right
  away, while the endpointer is still deciding whether the turn is over. When
  it is, the first words are already waiting and play instantly. If the user
  goes on talking, the speculation is thrown away without a trace.
* **Adaptive end of turn.** How long a pause must be depends on the words:
  "...and" waits, "...please." does not.
* **Streaming everywhere.** LLM tokens become speakable chunks; chunks become
  80 ms audio blocks while they are generated; G2P for the next sentence
  overlaps with audio of the current one.
* **Append-only memory.** The LLM never re-reads the conversation. Even an
  interruption is expressed as a note on the next message, not a rewrite.
* **Barge-in.** Talking over the assistant stops it within a few frames, and
  the model is told exactly what the user heard.

The engine is transport-agnostic: ``feed_audio()`` in, ``audio`` events (or an
attached sink) out. ``listen()`` wires up the local microphone and speakers.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from . import hardware, registry
from .audio import resample, to_float32
from .config import Config
from .events import EventBus
from .lang import get_language
from .llm import ChatMessage, LLMBackend
from .prosody import ProsodyAnalyzer
from .stt import STTBackend
from .text import SentenceChunker, StreamFilter, clean_for_tts, completeness, has_speakable
from .tools import Tool, ToolRegistry
from .tts import TTSBackend, VoiceCloningNotSupported
from .vad import FRAME, FRAME_MS, BargeInDetector, Endpointer, TurnEvent, VADFrontEnd, VADModel

log = logging.getLogger("sokhan")
SR = 16000
_END = object()


class State(str, Enum):
    LOADING = "loading"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


@dataclass
class _Chunk:                                 # one TTS text chunk on the output timeline
    text: str
    start: Optional[float] = None
    end: float = 0.0


@dataclass
class _Turn:
    id: int
    user: Optional[ChatMessage]
    speculative: bool
    utt: int = -1
    pause: int = -1
    t_user_end: float = 0.0
    speak: bool = True
    cancel: threading.Event = field(default_factory=threading.Event)
    committed: bool = False
    held: List[np.ndarray] = field(default_factory=list)
    pending_events: List[tuple] = field(default_factory=list)
    timeline: List[_Chunk] = field(default_factory=list)
    new_msgs: List[ChatMessage] = field(default_factory=list)
    spoken: str = ""
    llm_done: bool = False
    tts_done: bool = False
    history_done: bool = False
    audio_started: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    m: Dict[str, float] = field(default_factory=dict)


class Omni:
    """The realtime voice assistant engine.

    >>> omni = Omni()            # defaults: 4-bit models, CPU, auto-download
    >>> omni.start()             # load models (blocks until ready)
    >>> omni.listen()            # microphone + speakers
    >>> omni.run_forever()
    """

    def __init__(self, config: Optional[Config] = None, *, tools: Optional[List[Union[Tool, Callable]]] = None,
                 stt: Optional[STTBackend] = None, llm: Optional[LLMBackend] = None,
                 tts: Optional[TTSBackend] = None, vad: Optional[VADModel] = None, emotion_model=None):
        self.config = config or Config()
        c = self.config
        self.events = EventBus()
        self.hw = hardware.detect()
        self.plan = hardware.plan(c, self.hw)
        self.vad_model = vad or registry.create("vad", c)
        self.stt = stt or registry.create("stt", c)
        self.llm = llm or registry.create("llm", c)
        self.tts = tts or registry.create("tts", c)
        self.tools = ToolRegistry(tools)
        self.prosody = ProsodyAnalyzer(c.prosody, emotion_model)
        self.lang = get_language(c.language)
        self.history: List[ChatMessage] = []

        self._state = State.LOADING
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._load_error: Optional[BaseException] = None
        self._threads: List[threading.Thread] = []

        self._in_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=500)
        self._ctl_q: "queue.Queue[tuple]" = queue.Queue()
        self._stt_q: "queue.PriorityQueue" = queue.PriorityQueue()
        self._brain_q: "queue.Queue" = queue.Queue()
        self._front_q: "queue.Queue" = queue.Queue()
        self._voice_q: "queue.Queue" = queue.Queue()
        self._seq = 0
        self._partial_busy = False

        self._lock = threading.RLock()              # turn + output state
        self._hist_lock = threading.RLock()
        self._turn: Optional[_Turn] = None
        self._turn_ids = 0
        self._live = (0, -1)                        # (utterance, pause) the VAD is currently in
        self._commit_waiting: Optional[tuple] = None
        self._play_end = 0.0
        self._sinks: List[Any] = []
        self._devices: List[Any] = []
        self._pending_pause: Optional[tuple] = None
        self._note = ""
        self._image: Optional[str] = None
        self._phrase_cache: "collections.OrderedDict[str, np.ndarray]" = collections.OrderedDict()
        self._fillers: set = set()
        self._recent = collections.deque(maxlen=50)  # ~1.6 s of mic frames (barge-in pre-roll)
        self._partial_ms = 0.0
        self._level_n = 0
        self._compact_abort = threading.Event()
        self.metrics_log: collections.deque = collections.deque(maxlen=100)
        self._system = ChatMessage("system", "")

    # ================================================================== events
    def on(self, event: str, fn: Optional[Callable] = None):
        """Subscribe to an event (see ``sokhan.events.EVENTS``); works as a decorator too."""
        return self.events.on(event, fn)

    def off(self, event: str, fn: Optional[Callable] = None) -> None:
        self.events.off(event, fn)

    def _emit(self, event: str, **kw) -> None:
        self.events.emit(event, **kw)

    def _emit_turn(self, turn: _Turn, event: str, **kw) -> None:
        """Speculative turns keep their events until they are confirmed."""
        with turn.lock:
            if not turn.committed:
                turn.pending_events.append((event, kw))
                return
        self._emit(event, **kw)

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, s: State) -> None:
        with self._state_lock:
            if s == self._state or self._state == State.CLOSED:
                return
            self._state = s
        self._emit("state", state=s)

    @property
    def capabilities(self) -> Dict[str, bool]:
        """What the configured models can do (checked, not assumed)."""
        return {"voice_cloning": bool(getattr(self.tts, "supports_cloning", False)),
                "streaming_tts": bool(getattr(self.tts, "supports_streaming", False)),
                "vision": bool(self.config.vision.enabled and getattr(self.llm, "supports_images", False)),
                "tools": bool(self.tools), "barge_in": bool(self.config.turn.barge_in),
                "speculative_turns": bool(self.config.turn.speculative)}

    @property
    def output_sample_rate(self) -> int:
        return int(getattr(self.tts, "sample_rate", 24000))

    # ================================================================== lifecycle
    def start(self, block: bool = True, timeout: Optional[float] = None) -> "Omni":
        """Download (first run) and load all models. ``block=False`` returns at once;
        wait for the ``ready`` event or call ``wait_ready()``."""
        if self._threads:
            return self
        t = threading.Thread(target=self._load, name="sokhan-load", daemon=True)
        t.start()
        self._threads.append(t)
        if block:
            self.wait_ready(timeout)
        return self

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        ok = self._ready.wait(timeout)
        if self._load_error:
            raise RuntimeError(f"Sokhan failed to load: {self._load_error}") from self._load_error
        return ok

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._load_error is None

    def _progress(self, stage: str, fraction: float, message: str = "") -> None:
        self._emit("load_progress", stage=stage, fraction=float(fraction), message=message)

    def _estimated_ram_mb(self) -> int:
        c = self.config
        mb = 150                                   # VAD, STT (4-bit), buffers
        if c.llm.backend == "llama_cpp":
            mb += 3300 if "4b" in c.llm.model.lower() else 1800 if "2b" in c.llm.model.lower() else 2500
        if c.tts.backend == "pocket_tts":
            mb += 350 if c.tts.quant == "q4" else 650
        return mb

    def _load(self) -> None:
        c = self.config
        try:
            for n in self.plan.notes:
                self._emit("warning", message=n)
            if c.hardware.low_priority:
                hardware.lower_priority()
            if not all(type(b).__name__.startswith("Mock") for b in (self.stt, self.llm, self.tts)):
                hardware.check_memory(c, self._estimated_ram_mb(), "the configured models")
            if c.language != "fa" and type(self.tts).__name__ == "PocketTTS":
                self._emit("warning", message="the default voice (Pocket-TTS) speaks Persian only; use "
                                              f"Config.for_language({c.language!r}) for {c.language} speech")
            ctx = registry.LoadContext(c, self.plan, self._progress)
            errors: List[BaseException] = []

            def job(name: str, fn: Callable[[], None]) -> threading.Thread:
                def run():
                    try:
                        self._progress(name, 0.0, f"loading {name}")
                        fn()
                        self._progress(name, 1.0, f"{name} ready")
                    except BaseException as e:           # noqa: BLE001
                        log.exception("loading %s failed", name)
                        errors.append(e)
                t = threading.Thread(target=run, name=f"sokhan-load-{name}", daemon=True)
                t.start()
                return t

            ts = [job("vad", lambda: self.vad_model.load(ctx)), job("stt", lambda: self.stt.load(ctx)),
                  job("tts", lambda: self.tts.load(ctx)), job("llm", lambda: self.llm.load(ctx))]
            for t in ts:
                t.join()
            if errors:
                raise errors[0]
            self.frontend = VADFrontEnd(c.vad, self.vad_model)
            self.endpointer = Endpointer(c.vad, c.turn)
            self.barge = BargeInDetector(c.turn)
            self._system = ChatMessage("system", c.system_prompt(tools=list(self.tools.tools)))
            self._progress("warmup", 0.0, "warming up")
            warm = [threading.Thread(target=self._safe, args=(self.stt.warmup,), daemon=True),
                    threading.Thread(target=self._safe, args=(self.tts.warmup,), daemon=True)]
            for t in warm:
                t.start()
            self.llm.prefill([self._system], self.tools.specs())       # system prompt read once, forever
            for t in warm:
                t.join()
            self._progress("warmup", 1.0, "ready")
            for name, fn in (("audio", self._audio_loop), ("stt", self._stt_loop), ("brain", self._brain_loop),
                             ("tts-text", self._front_loop), ("tts-voice", self._voice_loop)):
                t = threading.Thread(target=fn, name=f"sokhan-{name}", daemon=True)
                t.start()
                self._threads.append(t)
            self._set_state(State.IDLE)
            self._ready.set()
            self._emit("ready")
            if c.turn.filler_after_ms > 0:
                threading.Thread(target=self._prefetch_fillers, daemon=True).start()
            if c.prompt.greeting:
                self.say(c.prompt.greeting)
        except BaseException as e:                      # noqa: BLE001
            self._load_error = e
            self._ready.set()
            self._emit("error", stage="load", error=e)

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            log.debug("warmup failed: %s", e)

    def close(self) -> None:
        """Stop everything and free the models."""
        if self._stop.is_set():
            return
        for dev in self._devices:
            try:
                dev.stop()
            except Exception:
                pass
        self._stop.set()
        self._compact_abort.set()
        with self._lock:
            if self._turn:
                self._turn.cancel.set()
        self._brain_q.put(None)
        self._front_q.put(None)
        self._voice_q.put(None)
        self._stt_q.put((0, 0, None))
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=3.0)
        for b in (self.stt, self.llm, self.tts):
            try:
                b.close()
            except Exception:
                pass
        self._set_state(State.CLOSED)

    stop = close

    def __enter__(self) -> "Omni":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def run_forever(self) -> None:
        """Block until Ctrl+C (then close)."""
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    # ================================================================== audio in / out
    def listen(self, input_device=None, output_device=None):
        """Talk to the assistant through this computer's microphone and speakers."""
        from .audio import LocalAudio
        self.wait_ready()
        dev = LocalAudio(self, input_device, output_device).start()
        self._devices.append(dev)
        return dev

    def feed_audio(self, pcm, sr: int = SR) -> None:
        """Push microphone audio (any rate, int16 or float32, mono or stereo). Thread-safe."""
        if not self._ready.is_set() or self._stop.is_set():
            return
        x = to_float32(pcm)
        if sr != SR:
            x = resample(x, sr, SR)
        try:
            self._in_q.put_nowait(x)
        except queue.Full:                         # overloaded: drop the oldest, never block the callback
            try:
                self._in_q.get_nowait()
                self._in_q.put_nowait(x)
            except (queue.Empty, queue.Full):
                pass

    def attach_sink(self, sink) -> None:
        """Send audio to an object with ``write(audio, sr)`` / ``flush()`` (+ optional
        ``pending_seconds()`` and ``level_db()`` for accurate echo handling)."""
        if sink is not None and sink not in self._sinks:
            self._sinks.append(sink)

    def detach_sink(self, sink) -> None:
        if sink in self._sinks:
            self._sinks.remove(sink)

    # ================================================================== public API
    def send_text(self, text: str, image=None, speak: bool = True) -> int:
        """Treat ``text`` as if the user had said it. Returns the turn id."""
        self.wait_ready()
        if image is not None:
            self.set_image(image)
        self._interrupt("new_input")
        turn = self._new_turn(text, audio=None, speculative=False, t_end=time.monotonic(), speak=speak)
        return turn.id if turn else -1

    def ask(self, text: str, speak: bool = False, timeout: Optional[float] = 120.0, image=None) -> str:
        """Blocking text chat: returns the assistant's reply (spoken too with ``speak=True``)."""
        self.wait_ready()
        if image is not None:
            self.set_image(image)
        self._interrupt("new_input")
        turn = self._new_turn(text, audio=None, speculative=False, t_end=time.monotonic(), speak=speak)
        if turn is None:
            return ""
        turn.finished.wait(timeout)
        return turn.spoken.strip()

    chat = ask

    def say(self, text: str, remember: bool = True) -> None:
        """Speak ``text`` verbatim (greetings, notifications)."""
        self.wait_ready()
        self._interrupt("say")
        with self._lock:
            self._turn_ids += 1
            turn = _Turn(self._turn_ids, None, speculative=False, committed=True, t_user_end=time.monotonic())
            turn.llm_done = turn.history_done = True
            turn.spoken = text
            self._turn = turn
        if remember:
            with self._hist_lock:
                self.history.append(ChatMessage("assistant", text))
        self._set_state(State.SPEAKING)
        chunker = self._chunker()
        for part in chunker.push(text) + chunker.flush():
            self._speak(turn, part)
        self._front_q.put((turn, _END))

    def interrupt(self) -> None:
        """Stop talking / thinking right now."""
        self._interrupt("api")

    def reset(self) -> None:
        """Forget the conversation."""
        self._interrupt("reset")
        with self._hist_lock:
            self.history.clear()
        self._note = ""
        self._brain_q.put(("prefill",))

    reset_conversation = reset

    def set_image(self, image) -> None:
        """Attach an image to the next user turn (vision models only)."""
        if not self.config.vision.enabled or not getattr(self.llm, "supports_images", False):
            self._emit("warning", message="image ignored: vision is off or the LLM has no vision projector")
            return
        from .vision import to_data_url
        v = self.config.vision
        self._image = to_data_url(image, v.max_side, v.jpeg_quality)

    def clear_image(self) -> None:
        self._image = None

    def clone_voice(self, sample, sr: Optional[int] = None) -> None:
        """Speak with the voice in ``sample`` (a ~5 s WAV path, or a numpy array + ``sr``).

        Raises ``VoiceCloningNotSupported`` when the TTS model cannot clone -
        check ``omni.capabilities["voice_cloning"]`` first.
        """
        if not self.capabilities["voice_cloning"]:
            raise VoiceCloningNotSupported(f"the TTS backend {type(self.tts).__name__!r} cannot clone voices")
        self.tts.set_voice(sample, sr)
        self._phrase_cache.clear()

    def add_tool(self, fn: Union[Tool, Callable]) -> Tool:
        """Register a tool. Prefer passing ``tools=`` to the constructor: adding one later
        changes the system prompt, which the LLM then has to re-read once."""
        t = self.tools.add(fn)
        if self._ready.is_set():
            self._brain_q.put(("prefill",))
        return t

    # ================================================================== audio loop (VAD)
    def _audible(self) -> bool:
        if time.monotonic() < self._play_end + self.config.turn.echo_tail_ms / 1000:
            return True
        return any(getattr(s, "pending_seconds", lambda: 0.0)() > 0.02 for s in self._sinks)

    def _playback_db(self) -> Optional[float]:
        for s in self._sinks:
            fn = getattr(s, "level_db", None)
            if fn:
                db = fn()
                if db is not None:
                    return db
        return None

    def _audio_loop(self) -> None:
        buf = np.zeros(0, np.float32)
        while not self._stop.is_set():
            self._drain_ctl()
            try:
                chunk = self._in_q.get(timeout=0.03)
                buf = np.concatenate([buf, chunk]) if len(buf) else chunk
                while len(buf) >= FRAME:
                    frame, buf = buf[:FRAME], buf[FRAME:]
                    self._process_frame(frame)
            except queue.Empty:
                pass
            except Exception:
                log.exception("audio frame failed")
            self._tick()

    def _drain_ctl(self) -> None:
        while True:
            try:
                msg = self._ctl_q.get_nowait()
            except queue.Empty:
                return
            if msg[0] == "end_ms":
                _, utt, pause, ms = msg
                self._handle(self.endpointer.set_end_silence(utt, pause, ms))

    def _process_frame(self, frame: np.ndarray) -> None:
        info = self.frontend(frame)
        self._recent.append(frame)
        self._level_n += 1
        if self._level_n >= 3:
            self._level_n = 0
            self._emit("level", db=info.db, speech=info.smooth)
        t = self.config.turn
        if self._audible():
            if t.half_duplex or not t.barge_in:
                return
            if self.barge.update(info, self._playback_db()):
                need = int((t.barge_in_min_ms + 400) / FRAME_MS)
                pre = np.concatenate(list(self._recent)[-need:])
                self.barge.reset()
                self._interrupt("barge_in")
                self._handle([self.endpointer.force_start(pre)])
            return
        self.barge.reset()
        self._handle(self.endpointer.process(frame, info))
        if self.endpointer.in_speech and not self.endpointer.paused and self.config.stt.partials:
            self._partial_ms += FRAME_MS
            if self._partial_ms >= self.config.stt.partial_interval_ms and not self._partial_busy:
                self._partial_ms = 0.0
                self._partial_busy = True
                self._stt(2, "partial", self.endpointer.snapshot(), self.endpointer.utt, -1)

    def _handle(self, events: List[TurnEvent]) -> None:
        for ev in events:
            if ev.kind == "start":
                self._interrupt("user_speech")
                self._partial_ms = 0.0
                self._live = (ev.utt, -1)
                self._set_state(State.LISTENING)
                self._emit("speech_start")
            elif ev.kind == "pause":
                with self._lock:
                    self._live = (ev.utt, ev.pause)
                self._stt(0, "pause", ev.audio, ev.utt, ev.pause, time.monotonic() - ev.silence_ms / 1000)
            elif ev.kind == "resume":
                with self._lock:
                    self._live = (ev.utt, -1)
                    t = self._turn
                    if t is not None and not t.committed and t.utt == ev.utt:
                        self._drop(t)
            elif ev.kind == "end":
                self._emit("speech_end", duration_ms=ev.speech_ms)
                t_end = time.monotonic() - ev.silence_ms / 1000
                with self._lock:
                    self._live = (0, -1)
                    t = self._turn
                    if t is not None and not t.committed and (t.utt, t.pause) == (ev.utt, ev.pause):
                        self._commit(t)
                        continue
                    if (ev.utt, ev.pause) == self._pending_pause:
                        self._commit_waiting = (ev.utt, ev.pause)     # its STT result is on the way
                        self._set_state(State.THINKING)
                        continue
                self._set_state(State.THINKING)
                self._stt(0, "final", ev.audio, ev.utt, ev.pause, t_end)

    def _tick(self) -> None:
        """Housekeeping on the audio thread: finish turns whose audio has played out."""
        with self._lock:
            t = self._turn
            if t is None or not t.committed:
                return
            fa = self.config.turn.filler_after_ms
            if (fa > 0 and t.speak and t.user is not None and not t.audio_started and "filler" not in t.m
                    and time.monotonic() - t.m.get("t_commit", time.monotonic()) > fa / 1000):
                t.m["filler"] = 1                  # nothing to say yet: a natural "hmm..." bridges the gap
                fillers = [a for p, a in self._phrase_cache.items() if p in self._fillers]
                if fillers:
                    self._play(t, (fillers[t.id % len(fillers)], _Chunk("")))
            if not (t.llm_done and (t.tts_done or not t.speak)) or self._audible():
                return
            self._turn = None
        self._finish(t, interrupted=False)
        if self._state != State.LISTENING:
            self._set_state(State.IDLE)

    # ================================================================== STT
    def _stt(self, prio: int, kind: str, audio, utt: int, pause: int, t_end: float = 0.0) -> None:
        self._seq += 1
        if kind == "pause":
            with self._lock:
                self._pending_pause = (utt, pause)
        self._stt_q.put((prio, self._seq, (kind, audio, utt, pause, t_end)))

    def _stt_loop(self) -> None:
        while not self._stop.is_set():
            _, _, job = self._stt_q.get()
            if job is None:
                return
            kind, audio, utt, pause, t_end = job
            try:
                if kind == "partial":
                    try:
                        if self._live[0] == utt:
                            r = self.stt.transcribe(audio, SR)
                            if r.text and self._live[0] == utt:
                                self._emit("partial_transcript", text=r.text)
                    finally:
                        self._partial_busy = False
                    continue
                t0 = time.perf_counter()
                r = self.stt.transcribe(audio, SR)
                stt_ms = (time.perf_counter() - t0) * 1000
                text = r.text.strip()
                if kind == "pause":
                    self._on_pause_text(text, audio, utt, pause, t_end, stt_ms)
                else:
                    if not self._meaningful(text):
                        self._set_state(State.IDLE)
                        continue
                    turn = self._new_turn(text, audio, speculative=False, t_end=t_end, utt=utt, pause=pause)
                    if turn:
                        turn.m["stt_ms"] = stt_ms
            except Exception as e:                               # noqa: BLE001
                log.exception("STT failed")
                self._emit("error", stage="stt", error=e)

    def _meaningful(self, text: str) -> bool:
        return len(text) >= 2 and has_speakable(text)

    def _on_pause_text(self, text: str, audio, utt: int, pause: int, t_end: float, stt_ms: float) -> None:
        t = self.config.turn
        ok = self._meaningful(text)
        if ok:
            c = completeness(text, self.config.language)
            if c < 0.9 and self.config.prosody.enabled and len(audio) > SR // 2:
                slope = self.prosody.analyze(audio, SR, text, update_baseline=False).end_slope_st
                if slope < -1.5 or slope > 1.5:          # clearly falling / rising end: sounds finished
                    c = max(c, 0.85)
            frac = min(1.0, max(0.0, (c - 0.15) / 0.6))       # 0.5 -> ~700 ms, >=0.75 -> end_silence_ms
            end_ms = t.end_silence_max_ms - frac * (t.end_silence_max_ms - t.end_silence_ms)
        else:
            end_ms = t.end_silence_max_ms * 1.5                  # noise: let a real end decide
        content = self._user_content(text, audio, update_baseline=False) if ok else ""
        turn = None
        with self._lock:                                         # atomic against the VAD thread
            if self._pending_pause == (utt, pause):
                self._pending_pause = None
            waiting = self._commit_waiting == (utt, pause)
            if not waiting and self._live != (utt, pause):
                return                                           # the user already went on talking
            self._commit_waiting = None
            if not ok:
                if waiting:
                    self._set_state(State.IDLE)
            elif t.speculative or waiting:
                turn = self._create_turn(text, content, speculative=not waiting, t_end=t_end,
                                         utt=utt, pause=pause)
        if turn:
            turn.m["stt_ms"] = stt_ms
            self._brain_q.put(turn)
        if not waiting:
            self._ctl_q.put(("end_ms", utt, pause, end_ms))

    # ================================================================== turns
    def _chunker(self) -> SentenceChunker:
        t = self.config.tts
        return SentenceChunker(t.first_chunk_min_chars, t.chunk_min_chars, t.chunk_max_chars,
                               t.first_chunk_max_words)

    def _user_content(self, text: str, audio: Optional[np.ndarray], update_baseline: bool) -> str:
        c = self.config
        if c.prosody.enabled and audio is not None and len(audio) / SR * 1000 >= c.prosody.min_utterance_ms:
            ft = self.prosody.analyze(audio, SR, text, update_baseline=update_baseline)
            cue = self.prosody.describe(ft) if c.prosody.inject else ""
            self._emit("prosody", features=ft, cue=cue)
            if cue:
                return f"{text}\n{cue}"
        return text

    def _new_turn(self, text: str, audio: Optional[np.ndarray], speculative: bool, t_end: float,
                  utt: int = -1, pause: int = -1, speak: bool = True) -> Optional[_Turn]:
        content = self._user_content(text, audio, update_baseline=True)
        with self._lock:
            turn = self._create_turn(text, content, speculative, t_end, utt, pause, speak)
        self._brain_q.put(turn)
        return turn

    def _create_turn(self, text: str, content: str, speculative: bool, t_end: float, utt: int = -1,
                     pause: int = -1, speak: bool = True) -> _Turn:
        """(caller holds ``self._lock``)"""
        if self._note and self.config.turn.interruption_note:
            content = f'[interrupted after: "{self._note}"]\n{content}'
        images = [self._image] if self._image else []
        self._turn_ids += 1
        turn = _Turn(self._turn_ids, ChatMessage("user", content, images), speculative, utt, pause,
                     t_user_end=t_end, speak=speak)
        turn.m["t0"] = time.monotonic()
        turn.pending_events.append(("transcript", {"text": text}))
        old = self._turn
        if old is not None:
            if old.committed:
                self._interrupt_locked("new_input")
            else:
                self._drop(old)
        self._turn = turn
        if not speculative:
            self._commit(turn)
        return turn

    def _commit(self, turn: _Turn) -> None:
        """The user's turn is really over: release everything the speculation prepared."""
        with self._lock:
            if turn.committed or turn.cancel.is_set():
                return
            with turn.lock:
                turn.committed = True
                events, turn.pending_events = turn.pending_events, []
            self._note = ""
            self._image = None
            turn.m["t_commit"] = time.monotonic()
            for ev, kw in events:
                self._emit(ev, **kw)
            held, turn.held = turn.held, []
            for blk in held:
                self._play(turn, blk)
            if not held:
                self._set_state(State.THINKING)
            if turn.llm_done:
                self._commit_history(turn)
                self._brain_q.put(("housekeep",))

    def _drop(self, turn: _Turn) -> None:
        """Silently discard a speculative turn (the user kept talking)."""
        turn.cancel.set()
        turn.held.clear()
        turn.finished.set()
        if self._turn is turn:
            self._turn = None

    def _commit_history(self, turn: _Turn) -> None:
        with turn.lock:
            if turn.history_done or turn.user is None:
                return
            turn.history_done = True
            msgs = [turn.user] + list(turn.new_msgs)
        with self._hist_lock:
            self.history.extend(msgs)

    def _finish(self, turn: _Turn, interrupted: bool, heard: str = "") -> None:
        text = heard if interrupted else turn.spoken.strip()
        if interrupted:
            turn.spoken = heard
        self._emit("response_done", text=text, interrupted=interrupted)
        turn.finished.set()

    # ================================================================== interruption
    def _heard(self, turn: _Turn, now: float) -> str:
        words: List[str] = []
        for ch in turn.timeline:
            if ch.start is None or ch.start >= now:
                continue
            w = ch.text.split()
            if ch.end <= now:
                words += w
            else:
                words += w[: int(len(w) * (now - ch.start) / max(1e-3, ch.end - ch.start))]
        return " ".join(words)

    def _interrupt(self, reason: str) -> None:
        with self._lock:
            self._interrupt_locked(reason)

    def _interrupt_locked(self, reason: str) -> None:
        turn = self._turn
        if turn is None:
            return
        if not turn.committed:
            self._drop(turn)
            return
        now = time.monotonic()
        heard = self._heard(turn, now)
        full = turn.spoken.strip()
        turn.cancel.set()
        self._turn = None
        self._play_end = 0.0
        for s in self._sinks:
            try:
                s.flush()
            except Exception:
                pass
        self._emit("audio_flush")
        if turn.user is not None and heard.strip() != full and (heard or turn.audio_started):
            self._note = heard.strip() or "..."
        self._compact_abort.set()
        self._emit("interrupted", reason=reason, heard=heard)
        self._finish(turn, interrupted=True, heard=heard)
        if reason not in ("user_speech", "barge_in"):
            self._set_state(State.IDLE)

    # ================================================================== brain (LLM)
    def _brain_loop(self) -> None:
        while not self._stop.is_set():
            item = self._brain_q.get()
            if item is None:
                return
            try:
                if isinstance(item, _Turn):
                    self._run_turn(item)
                elif item[0] == "housekeep":
                    self._housekeep()
                elif item[0] == "prefill":
                    self._system = ChatMessage("system", self.config.system_prompt(tools=list(self.tools.tools)))
                    with self._hist_lock:
                        msgs = [self._system] + list(self.history)
                    self.llm.prefill(msgs, self.tools.specs())
            except Exception as e:                           # noqa: BLE001
                log.exception("brain job failed")
                self._emit("error", stage="llm", error=e)
                if isinstance(item, _Turn):
                    with self._lock:
                        item.llm_done = item.tts_done = True
                    self._front_q.put((item, _END))

    def _run_turn(self, turn: _Turn) -> None:
        if turn.cancel.is_set():
            return
        c = self.config
        with self._hist_lock:
            msgs = [self._system] + list(self.history) + [turn.user]
        specs = self.tools.specs()
        chunker = self._chunker()
        rounds = 0
        t0 = time.monotonic()
        self._emit_turn(turn, "response_start", turn_id=turn.id)
        while True:
            filt = StreamFilter(start_in_think=c.llm.thinking)
            raw, calls = "", []
            gen = self.llm.stream(msgs, turn.cancel, specs)
            try:
                for piece in gen:
                    if turn.cancel.is_set():
                        break
                    if "llm_ttft_ms" not in turn.m:
                        turn.m["llm_ttft_ms"] = (time.monotonic() - t0) * 1000
                    if isinstance(piece, dict):
                        calls.append(piece)
                        continue
                    raw += piece
                    for kind, val in filt.feed(piece):
                        self._on_llm(turn, kind, val, chunker, calls)
            finally:
                gen.close()
            if not turn.cancel.is_set():
                for kind, val in filt.finish():
                    self._on_llm(turn, kind, val, chunker, calls)
            reply = ChatMessage("assistant", raw)
            self.llm.bind_reply(reply)
            with turn.lock:
                turn.new_msgs.append(reply)
            if calls and not turn.cancel.is_set() and rounds < 3 and self.tools:
                rounds += 1
                results = []
                for call in calls:
                    res = self.tools.call(call["name"], call.get("arguments", {}))
                    self._emit_turn(turn, "tool_call", name=call["name"], arguments=call.get("arguments", {}),
                                    result=res)
                    results.append(ChatMessage("tool", res))
                with turn.lock:
                    turn.new_msgs.extend(results)
                msgs = msgs + [reply] + results
                continue
            break
        if not turn.cancel.is_set():
            for part in chunker.flush():
                self._speak(turn, part)
        with self._lock:
            turn.llm_done = True
            committed = turn.committed
        self._front_q.put((turn, _END))
        if committed:
            self._commit_history(turn)
            self._housekeep()

    def _on_llm(self, turn: _Turn, kind: str, val, chunker: SentenceChunker, calls: list) -> None:
        if kind == "tool":
            calls.append(val)
            return
        turn.spoken += val
        self._emit_turn(turn, "response_delta", text=val)
        for part in chunker.push(val):
            self._speak(turn, part)

    def _speak(self, turn: _Turn, text: str) -> None:
        if not turn.speak or turn.cancel.is_set():
            return
        clean = clean_for_tts(text, self.config.language)
        if has_speakable(clean):
            if "first_chunk_ms" not in turn.m:
                turn.m["first_chunk_ms"] = (time.monotonic() - turn.m.get("t0", time.monotonic())) * 1000
            self._front_q.put((turn, clean))

    def _housekeep(self) -> None:
        """Idle-time work on the LLM thread: checkpoint, and compact long histories."""
        try:
            self.llm.checkpoint()
        except Exception as e:
            log.debug("checkpoint failed: %s", e)
        budget = self.config.llm.history_tokens
        with self._hist_lock:
            sizes = [self.llm.count_tokens(m) for m in self.history]
            if sum(sizes) <= budget:
                return
            keep, acc = [], 0
            for m, n in zip(reversed(self.history), reversed(sizes)):
                if acc + n > budget // 2 and keep:
                    break
                keep.insert(0, m)
                acc += n
            while keep and keep[0].role != "user":
                keep.pop(0)
            self.history[:] = keep
            msgs = [self._system] + keep
        self._compact_abort.clear()
        log.info("compacting history to %d messages", len(keep))
        self.llm.prefill(msgs, self.tools.specs(), cancel=self._compact_abort)

    # ================================================================== TTS pipeline
    def _front_loop(self) -> None:
        """Text front-end (normalisation, G2P) - runs ahead of the acoustic model."""
        while not self._stop.is_set():
            item = self._front_q.get()
            if item is None:
                self._voice_q.put(None)
                return
            turn, text = item
            if text is _END or turn.cancel.is_set():
                self._voice_q.put(item)
                continue
            try:
                cached = self._phrase_cache.get(text) if self.config.tts.cache_phrases else None
                plan = ("cached", cached) if cached is not None else ("plan", self.tts.prepare(text))
                self._voice_q.put((turn, (text, plan)))
            except Exception as e:                            # noqa: BLE001
                log.exception("TTS front-end failed")
                self._emit("error", stage="tts", error=e)

    def _voice_loop(self) -> None:
        while not self._stop.is_set():
            item = self._voice_q.get()
            if item is None:
                return
            turn, payload = item
            if payload is _END:
                with self._lock:
                    turn.tts_done = True
                continue
            if turn.cancel.is_set():
                continue
            text, (kind, plan) = payload
            chunk = _Chunk(text)
            with self._lock:
                turn.timeline.append(chunk)
            try:
                t0 = time.monotonic()
                if kind == "cached":
                    self._output(turn, plan, chunk)
                    continue
                keep = [] if (self.config.tts.cache_phrases and len(text) <= 48) else None
                for blk in self.tts.stream(plan, turn.cancel):
                    if turn.cancel.is_set():
                        break
                    if "tts_first_ms" not in turn.m:
                        turn.m["tts_first_ms"] = (time.monotonic() - t0) * 1000
                    if keep is not None:
                        keep.append(blk)
                    self._output(turn, blk, chunk)
                if keep and not turn.cancel.is_set():
                    self._phrase_cache[text] = np.concatenate(keep)
                    while len(self._phrase_cache) > 64:
                        self._phrase_cache.popitem(last=False)
            except Exception as e:                            # noqa: BLE001
                log.exception("TTS failed")
                self._emit("error", stage="tts", error=e)

    def _output(self, turn: _Turn, blk: np.ndarray, chunk: _Chunk) -> None:
        with self._lock:
            if turn.cancel.is_set():
                return
            if not turn.committed:
                turn.held.append((blk, chunk))
                return
            self._play(turn, (blk, chunk))

    def _play(self, turn: _Turn, item) -> None:
        blk, chunk = item
        sr = self.output_sample_rate
        now = time.monotonic()
        start = max(now, self._play_end)
        self._play_end = start + len(blk) / sr
        if chunk.start is None:
            chunk.start = start
        chunk.end = self._play_end
        first = not turn.audio_started
        turn.audio_started = True
        for s in self._sinks:
            try:
                s.write(blk, sr)
            except Exception:
                log.exception("audio sink failed")
        self._emit("audio", audio=blk, sr=sr)
        if first:
            self._set_state(State.SPEAKING)
            m = turn.m
            m["latency_ms"] = (now - turn.t_user_end) * 1000 if turn.t_user_end else 0.0
            report = {k: round(v, 1) for k, v in m.items() if k.endswith("_ms")}
            report["speculative"] = turn.speculative
            self.metrics_log.append(report)
            self._emit("metrics", **report)

    def _prefetch_fillers(self) -> None:
        for p in self.config.prompt.fillers or self.lang.fillers:
            try:
                clean = clean_for_tts(p, self.config.language)
                self._phrase_cache[clean] = self.tts.synthesize(clean)
                self._fillers.add(clean)
            except Exception:
                pass


OmniEngine = Omni                                    # backwards-compatible name
