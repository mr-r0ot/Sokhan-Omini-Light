"""OmniEngine — the fused realtime core.

    mic ─► VAD/endpointer ─► STT ─┐
             │  ▲                 ├─► prosody cues ─► LLM (streaming) ─► chunker ─► TTS ─► speaker
       barge-in detector ◄────────┘                        ▲                               │
             └──── cancel LLM + TTS + flush audio ─────────┴───────────────────────────────┘

Everything is event driven and I/O-agnostic:
    engine.feed_audio(pcm, sr)          # push microphone / telephony audio in
    engine.on("audio_out", cb)          # pull synthesized audio out (or attach LocalAudio)
"""
from __future__ import annotations

import collections
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import hardware, models
from .audio_io import resample, to_float32
from .config import Config
from .llm import ChatMessage, LLMBackend, create_llm
from .prosody import ProsodyAnalyzer
from .stt import STTBackend, create_stt
from .text import SentenceChunker, StreamFilter, clean_for_tts, completeness, has_speakable
from .tools import Tool, ToolRegistry
from .tts import CachedTTS, TTSBackend, create_tts
from .vad import (FRAME, BargeInDetector, Endpointer, VADFrontEnd, load_vad_model, rms_db)

log = logging.getLogger("sokhan.engine")
SR = 16000


class State(str, Enum):
    LOADING = "loading"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class Utterance:
    audio: Optional[np.ndarray]
    text: Optional[str] = None
    seq: int = -1
    t_speech_end: float = 0.0
    typed: bool = False


@dataclass
class _Chunk:
    turn: int
    text: str
    start: float
    end: float
    filler: bool = False


@dataclass
class _TTSJob:
    turn: int
    text: str = ""
    end: bool = False
    filler: bool = False


class OmniEngine:
    def __init__(self, config: Optional[Config] = None, *, stt: Optional[STTBackend] = None,
                 llm: Optional[LLMBackend] = None, tts: Optional[TTSBackend] = None,
                 tools: Optional[List[Tool]] = None, vad_model=None, emotion_model=None):
        self.config = config or Config()
        c = self.config
        self.hw = hardware.detect_hardware()
        self.plan = hardware.plan_resources(c, self.hw)
        self.stt = stt or create_stt(c)
        self.llm = llm or create_llm(c)
        self._tts_backend = tts or create_tts(c)
        self.tts = CachedTTS(self._tts_backend, c.tts.cache_max_mb, c.tts.cache_phrases)
        self.tools = ToolRegistry(tools)
        self.prosody = ProsodyAnalyzer(c.prosody, emotion_model)
        self._vad_model_override = vad_model
        self.frontend: Optional[VADFrontEnd] = None
        self.endpointer = Endpointer(c.vad, c.audio.frame_ms)
        self.barge = BargeInDetector(c.barge_in, c.vad, c.audio.frame_ms)

        self._state = State.LOADING
        self._handlers: Dict[str, List[Callable]] = collections.defaultdict(list)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._threads: List[threading.Thread] = []

        self._in_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self._ctl_q: "queue.Queue[tuple]" = queue.Queue()
        self._stt_q: "queue.PriorityQueue" = queue.PriorityQueue()
        self._turn_q: "queue.Queue[Optional[Utterance]]" = queue.Queue()
        self._tts_q: "queue.Queue[Optional[_TTSJob]]" = queue.Queue()
        self._counter = 0

        self._turn_lock = threading.RLock()
        self._turn_id = 0
        self._cancel = threading.Event()
        self._chunks: List[_Chunk] = []
        self._play_end = 0.0
        self._playback = None
        self._recent = collections.deque(maxlen=40)          # ~1.3 s of frames for barge-in pre-roll
        self._partial_ms = 0.0
        self._early: Dict[int, Any] = {}
        self._early_events: Dict[int, threading.Event] = {}
        self._frames_since_level = 0

        self.history: List[ChatMessage] = []
        self._hist_lock = threading.Lock()
        self._assistant_msgs: Dict[int, ChatMessage] = {}
        self._cancel_info: Dict[int, str] = {}
        self._carry_text = ""
        self._image: Optional[str] = None
        self._turn_metrics: Dict[str, Any] = {}
        self.metrics_log: collections.deque = collections.deque(maxlen=50)
        self._overload_warned = False
        self._loaded_error: Optional[BaseException] = None

    # ================================================================= events
    def on(self, event: str, cb: Callable) -> Callable[[], None]:
        self._handlers[event].append(cb)
        return lambda: self._handlers[event].remove(cb) if cb in self._handlers[event] else None

    def emit(self, event: str, **kw) -> None:
        for cb in list(self._handlers.get(event, ())):
            try:
                cb(**kw)
            except Exception:
                log.exception("handler for %s failed", event)

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, s: State) -> None:
        if s != self._state:
            self._state = s
            self.emit("state", state=s)

    # ================================================================= lifecycle
    def start(self, block: bool = False, warmup: bool = True) -> "OmniEngine":
        t = threading.Thread(target=self._load, args=(warmup,), name="sokhan-load", daemon=True)
        t.start()
        self._threads.append(t)
        if block:
            self.wait_ready()
        return self

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        ok = self._ready.wait(timeout)
        if self._loaded_error:
            raise RuntimeError(f"engine failed to load: {self._loaded_error}") from self._loaded_error
        return ok

    def _progress(self, label: str, frac: float) -> None:
        self.emit("load_progress", label=label, fraction=frac)

    def estimated_ram_mb(self) -> int:
        c = self.config
        if c.stt.backend == "mock" and c.llm.backend == "mock" and c.tts.backend == "mock":
            return 10
        ram = 30 + (900 if c.stt.model == "koochik" else 350) + 700
        ram += 3300 if c.llm.quantized else 8600
        if c.vision.enabled:
            ram += 900
        return ram + 200

    def _load(self, warmup: bool) -> None:
        c = self.config
        try:
            for n in self.plan.notes:
                self.emit("warning", message=n)
            if c.hardware.lower_worker_priority:
                hardware.lower_priority()
            hardware.check_memory(c, self.estimated_ram_mb(), "the configured models", self.hw)
            errors: List[BaseException] = []

            def run(name, fn):
                try:
                    self._progress(name, 0.0)
                    fn()
                    self._progress(name, 1.0)
                except BaseException as e:            # noqa: BLE001
                    log.exception("loading %s failed", name)
                    errors.append(e)

            def load_vad():
                path = None
                if self._vad_model_override is not None:
                    model = self._vad_model_override
                else:
                    if c.vad.backend != "energy":
                        try:
                            path = models.ensure_vad(c, self._progress)
                        except Exception as e:
                            self.emit("warning", message=f"Silero VAD unavailable ({e}); using energy VAD")
                    model = load_vad_model(c.vad, path)
                self.frontend = VADFrontEnd(c.vad, model)

            jobs = [("vad", load_vad),
                    ("stt", lambda: self.stt.load(self.plan, self._progress)),
                    ("tts", lambda: self._tts_backend.load(self.plan, self._progress)),
                    ("llm", lambda: self.llm.load(self.plan, self._progress))]
            ts = [threading.Thread(target=run, args=j, daemon=True) for j in jobs]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            if errors:
                raise errors[0]
            for name in ("vad", "stt", "tts", "llm"):
                pass
            self._spawn_workers()
            if warmup:
                self.stt.warmup()
                self.tts.synth("سلام", cache=False)
                self.llm.warmup(self._system_prompt())
            self._set_state(State.IDLE)
            self._ready.set()
            self.emit("ready")
            threading.Thread(target=self._prefetch_phrases, daemon=True, name="sokhan-prefetch").start()
        except BaseException as e:                    # noqa: BLE001
            self._loaded_error = e
            self._ready.set()
            self.emit("error", where="load", error=e)

    def _prefetch_phrases(self) -> None:
        p = self.config.prompt
        phrases = list(p.fillers) + ([p.greeting] if p.greeting else [])
        for ph in phrases:
            if self._stop.is_set():
                return
            try:
                self.tts.synth(clean_for_tts(ph))
            except Exception:
                log.debug("prefetch failed for %r", ph)
        if p.greeting:
            self.say(p.greeting)

    def _spawn_workers(self) -> None:
        for name, fn in (("vad", self._vad_loop), ("stt", self._stt_loop),
                         ("brain", self._brain_loop), ("tts", self._tts_loop)):
            t = threading.Thread(target=fn, name=f"sokhan-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def close(self) -> None:
        self._stop.set()
        self._cancel.set()
        for q_, item in ((self._turn_q, None), (self._tts_q, None)):
            q_.put(item)
        self._stt_q.put((0, 0, None))
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2.0)
        for b in (self.stt, self.llm, self._tts_backend):
            try:
                b.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start(block=True)

    def __exit__(self, *a):
        self.close()

    def run_forever(self) -> None:
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    # ================================================================= public API
    def feed_audio(self, pcm, sr: int = SR) -> None:
        """Thread-safe. float32 [-1,1] or int16, any chunk size, any rate, mono/stereo."""
        if self._state == State.LOADING or self._stop.is_set():
            return
        x = to_float32(pcm)
        if sr != SR:
            x = resample(x, sr, SR)
        try:
            self._in_q.put_nowait(x)
        except queue.Full:
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
            self._in_q.put_nowait(x)

    def attach_playback(self, playback) -> None:
        """Local speaker path (see audio_io.LocalAudio)."""
        self._playback = playback

    def send_text(self, text: str, image=None) -> None:
        """Skip speech input: treat ``text`` as if the user said it."""
        if image is not None:
            self.set_image(image)
        if self._state in (State.SPEAKING, State.THINKING):
            self._interrupt("typed")
        self._turn_q.put(Utterance(None, text, typed=True, t_speech_end=time.monotonic()))
        self._set_state(State.THINKING)

    def say(self, text: str, add_to_history: bool = True) -> None:
        """Speak ``text`` verbatim (greetings, notifications)."""
        with self._turn_lock:
            turn = self._new_turn()
        if add_to_history:
            with self._hist_lock:
                msg = ChatMessage("assistant", text)
                self.history.append(msg)
                self._assistant_msgs[turn] = msg
        ch = SentenceChunker(self.config.tts.first_chunk_min_chars, self.config.tts.chunk_min_chars,
                             self.config.tts.chunk_max_chars)
        for part in ch.push(text) + ch.flush():
            self._tts_q.put(_TTSJob(turn, part))
        self._tts_q.put(_TTSJob(turn, end=True))
        self._turn_metrics = {"t0": time.monotonic(), "kind": "say"}

    def interrupt(self) -> None:
        self._interrupt("api")

    def set_image(self, image) -> None:
        if not self.config.vision.enabled:
            self.emit("warning", message="vision is disabled in the config; image ignored")
            return
        from .vision import to_data_url
        v = self.config.vision
        self._image = to_data_url(image, v.max_side, v.jpeg_quality)

    def clear_image(self) -> None:
        self._image = None

    def reset_conversation(self) -> None:
        self._interrupt("reset")
        with self._hist_lock:
            self.history.clear()
            self._assistant_msgs.clear()
        self._carry_text = ""
        try:
            self.llm.warmup(self._system_prompt())
        except Exception:
            pass

    def register_tool(self, tool: Tool) -> None:
        self.tools.add(tool)

    # ================================================================= VAD loop
    def _agent_audible(self) -> bool:
        return time.monotonic() < self._play_end + self.config.barge_in.echo_tail_ms / 1000.0

    def _playback_db(self) -> Optional[float]:
        if self._playback is not None:
            return self._playback.level_db()
        if time.monotonic() < self._play_end:          # external sink: assume nominal level
            return -24.0
        return None

    def _vad_loop(self) -> None:
        buf = np.zeros(0, np.float32)
        while not self._stop.is_set():
            self._drain_ctl()
            try:
                chunk = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            buf = np.concatenate([buf, chunk]) if len(buf) else chunk
            backlog = self._in_q.qsize()
            if backlog > 60 and not self._overload_warned:
                self._overload_warned = True
                self.emit("warning", message="CPU overloaded: audio backlog building up (partials disabled)")
            elif backlog < 5:
                self._overload_warned = False
            while len(buf) >= FRAME:
                frame, buf = buf[:FRAME], buf[FRAME:]
                try:
                    self._process_frame(frame)
                except Exception:
                    log.exception("vad frame failed")

    def _drain_ctl(self) -> None:
        while True:
            try:
                kind, seq = self._ctl_q.get_nowait()
            except queue.Empty:
                return
            if kind == "early_ok":
                self._handle_events(self.endpointer.finalize(seq))

    def _process_frame(self, frame: np.ndarray) -> None:
        fe = self.frontend
        if fe is None:
            return
        info = fe(frame)
        self._recent.append(frame)
        self._frames_since_level += 1
        if self._frames_since_level >= 3:
            self._frames_since_level = 0
            self.emit("level", db=info.rms_db, speech=info.smooth)

        bi = self.config.barge_in
        if self._agent_audible():
            if bi.half_duplex or not bi.enabled:
                return
            if self.barge.update(info, self._playback_db()):
                need = int((bi.min_speech_ms + 500) / self.config.audio.frame_ms)
                pre = np.concatenate(list(self._recent)[-need:])
                self._barge_in(pre)
            return
        self.barge.reset()

        events = self.endpointer.process(frame, info)
        self._handle_events(events)
        if (self.endpointer.in_speech and self.config.stt.partials and not self._overload_warned
                and self._state == State.LISTENING):
            self._partial_ms += self.config.audio.frame_ms
            if self._partial_ms >= self.config.stt.partial_interval_ms and self._stt_q.qsize() == 0:
                self._partial_ms = 0.0
                self._submit_stt(2, "partial", self.endpointer.snapshot(), self.endpointer.seq)

    def _handle_events(self, events) -> None:
        for ev in events:
            if ev.kind == "start":
                self._on_speech_start()
            elif ev.kind == "possible_end":
                self._early_events[ev.seq] = threading.Event()
                self._submit_stt(0, "early", ev.audio, ev.seq)
            elif ev.kind == "end":
                self._partial_ms = 0.0
                t_end = time.monotonic() - ev.silence_ms / 1000.0
                self._turn_q.put(Utterance(ev.audio, None, ev.seq, t_end))
                self._set_state(State.THINKING)
                self.emit("speech_end", duration_ms=ev.speech_ms)

    def _on_speech_start(self) -> None:
        if self._state in (State.THINKING, State.SPEAKING):
            self._interrupt("user_speech")
        self._partial_ms = 0.0
        self._set_state(State.LISTENING)
        self.emit("speech_start")

    def _barge_in(self, preroll: np.ndarray) -> None:
        self._interrupt("barge_in")
        self.barge.reset()
        self._handle_events([self.endpointer.force_start(preroll)])

    # ================================================================= STT worker
    def _submit_stt(self, prio: int, kind: str, audio: np.ndarray, seq: int) -> None:
        self._counter += 1
        self._stt_q.put((prio, self._counter, (kind, audio, seq)))

    def _stt_loop(self) -> None:
        while not self._stop.is_set():
            _, _, job = self._stt_q.get()
            if job is None:
                break
            kind, audio, seq = job
            try:
                if kind == "partial":
                    if seq != self.endpointer.seq or not self.endpointer.in_speech:
                        continue
                    fn = getattr(self.stt, "try_transcribe", self.stt.transcribe)
                    res = fn(audio, SR)
                    if res is not None and res.text:
                        self.emit("partial", text=res.text)
                elif kind == "early":
                    res = self.stt.transcribe(audio, SR)
                    self._early[seq] = res
                    ev = self._early_events.get(seq)
                    if ev:
                        ev.set()
                    if self.endpointer.seq != seq:
                        continue                                  # user resumed meanwhile
                    thr = self.config.vad.early_complete_threshold
                    comp = completeness(res.text)
                    ok = comp >= thr
                    if not ok and comp >= 0.7 and self.config.prosody.enabled:
                        ft = self.prosody.analyze(audio, SR, res.text, update_baseline=False)
                        ok = ft.end_slope_st < -1.0 or ft.end_slope_st > 1.5
                    if ok:
                        self._ctl_q.put(("early_ok", seq))
            except Exception:
                log.exception("stt job failed")

    # ================================================================= brain
    def _new_turn(self) -> int:
        self._turn_id += 1
        self._cancel = threading.Event()
        self._chunks = []
        self._play_end = 0.0
        return self._turn_id

    def _system_prompt(self) -> str:
        return self.config.system_prompt() + self.tools.prompt_block()

    def _brain_loop(self) -> None:
        while not self._stop.is_set():
            utt = self._turn_q.get()
            if utt is None:
                break
            try:
                self._run_turn(utt)
            except Exception as e:                  # noqa: BLE001
                log.exception("turn failed")
                self.emit("error", where="turn", error=e)
                self._set_state(State.IDLE)

    def _run_turn(self, utt: Utterance) -> None:
        c = self.config
        m: Dict[str, Any] = {"t_speech_end": utt.t_speech_end or time.monotonic()}
        with self._turn_lock:
            turn = self._new_turn()
            cancel = self._cancel
        # ---------------- transcript
        text = utt.text
        if text is None:
            t = time.perf_counter()
            res = None
            ev = self._early_events.pop(utt.seq, None)
            if ev is not None:
                ev.wait(timeout=3.0)
                res = self._early.pop(utt.seq, None)
            self._early.clear()
            self._early_events.clear()
            if res is None:
                res = self.stt.transcribe(utt.audio, SR)
            text = res.text
            m["stt_ms"] = (time.perf_counter() - t) * 1000
        text = (text or "").strip()
        if cancel.is_set():
            return
        if len(text) < 2 or not has_speakable(text):
            self._set_state(State.IDLE)
            return
        if self._carry_text:
            text, self._carry_text = f"{self._carry_text} {text}", ""
        self.emit("transcript", text=text, final=True)
        # ---------------- prosody
        cue = ""
        if c.prosody.enabled and utt.audio is not None and len(utt.audio) / SR * 1000 >= c.prosody.min_utterance_ms:
            t = time.perf_counter()
            ft = self.prosody.analyze(utt.audio, SR, text)
            cue = self.prosody.describe(ft) if c.prosody.inject_into_prompt else ""
            m["prosody_ms"] = (time.perf_counter() - t) * 1000
            self.emit("prosody", features=ft, cue=cue)
        content = text + (("\n" + cue) if cue else "")
        images = [self._image] if (self._image and c.vision.enabled) else []
        self._image = None
        with self._hist_lock:
            self.history.append(ChatMessage("user", content, images))
            self._trim_history()
        self._turn_metrics = m
        self._respond(turn, cancel)

    def _trim_history(self) -> None:
        c = self.config.llm
        pairs_limit = c.history_max_turns * 2
        budget = int((c.n_ctx - c.max_tokens - 700) * 2.0)     # ~2 chars/token for Persian
        def size() -> int:
            return sum(len(m.content) for m in self.history)
        if len(self.history) <= pairs_limit and size() <= budget:
            return
        # drop in big steps so the KV-cache prefix stays valid for many turns afterwards
        while self.history and len(self.history) > max(2, pairs_limit // 2) or size() > budget // 2:
            if len(self.history) <= 1:
                break
            self.history.pop(0)
            while self.history and self.history[0].role != "user":
                self.history.pop(0)

    def _build_messages(self) -> List[ChatMessage]:
        with self._hist_lock:
            hist = list(self.history)
        last_img = max((i for i, m in enumerate(hist) if m.images), default=-1)
        msgs = [ChatMessage("system", self._system_prompt())]
        for i, m in enumerate(hist):
            msgs.append(ChatMessage(m.role, m.content, m.images if i == last_img else []))
        return msgs

    def _respond(self, turn: int, cancel: threading.Event) -> None:
        c = self.config
        m = self._turn_metrics
        rounds, spoken_raw = 0, ""
        filler_timer: Optional[threading.Timer] = None
        if c.prompt.fillers and c.prompt.filler_after_ms > 0:
            filler_timer = threading.Timer(c.prompt.filler_after_ms / 1000.0, self._maybe_filler, args=(turn,))
            filler_timer.daemon = True
            filler_timer.start()
        try:
            while True:
                self._set_state(State.THINKING) if not self._chunks else None
                filt = StreamFilter(start_in_think=c.llm.enable_thinking)
                chunker = SentenceChunker(c.tts.first_chunk_min_chars, c.tts.chunk_min_chars, c.tts.chunk_max_chars)
                calls: List[dict] = []
                raw = ""
                t_llm = time.perf_counter()
                n_chars = 0

                def handle(kind, val):
                    nonlocal raw, n_chars
                    if kind == "text":
                        raw += val
                        n_chars += len(val)
                        self.emit("llm_delta", text=val)
                        for part in chunker.push(val):
                            self._enqueue(turn, part, filler_timer)
                    else:
                        calls.append(val)

                for piece in self.llm.stream(self._build_messages(), cancel):
                    if cancel.is_set():
                        break
                    if "ttft_ms" not in m:
                        m["ttft_ms"] = (time.perf_counter() - t_llm) * 1000
                    for kind, val in filt.feed(piece):
                        handle(kind, val)
                if not cancel.is_set():
                    for kind, val in filt.finish():
                        handle(kind, val)
                    for part in chunker.flush():
                        self._enqueue(turn, part, filler_timer)
                dt = max(1e-3, time.perf_counter() - t_llm)
                m["llm_chars_per_s"] = n_chars / dt
                spoken_raw += raw
                if cancel.is_set():
                    break
                if calls and rounds < 3 and self.tools:
                    rounds += 1
                    blocks = "".join(f"\n<tool_call>\n{_dumps(cl)}\n</tool_call>" for cl in calls)
                    with self._hist_lock:
                        self.history.append(ChatMessage("assistant", raw.strip() + blocks))
                        for cl in calls:
                            self.emit("tool_call", name=cl["name"], arguments=cl["arguments"])
                            res = self.tools.call(cl["name"], cl["arguments"])
                            self.history.append(ChatMessage("user", f"<tool_response>\n{res}\n</tool_response>"))
                    continue
                break
        finally:
            if filler_timer:
                filler_timer.cancel()
        # ---------------- bookkeeping (history keeps only what was actually said when interrupted)
        with self._hist_lock:
            if cancel.is_set():
                spoken = self._cancel_info.pop(turn, "")
                if spoken:
                    msg = ChatMessage("assistant", spoken)
                    self.history.append(msg)
            elif spoken_raw.strip():
                msg = ChatMessage("assistant", spoken_raw.strip())
                self.history.append(msg)
                self._assistant_msgs[turn] = msg
        if not cancel.is_set():
            self.emit("assistant_done", text=spoken_raw.strip())
            self._tts_q.put(_TTSJob(turn, end=True))
            if not spoken_raw.strip():
                self._set_state(State.IDLE)

    def _enqueue(self, turn: int, part: str, filler_timer: Optional[threading.Timer]) -> None:
        if filler_timer:
            filler_timer.cancel()
        self.emit("assistant_text", text=part)
        self._tts_q.put(_TTSJob(turn, part))

    def _maybe_filler(self, turn: int) -> None:
        if turn != self._turn_id or self._chunks or self._cancel.is_set():
            return
        cached = [p for p in self.config.prompt.fillers if self.tts.get(clean_for_tts(p)) is not None]
        if cached:
            self._tts_q.put(_TTSJob(turn, random.choice(cached), filler=True))

    # ================================================================= TTS worker
    def _tts_loop(self) -> None:
        tail = self.config.barge_in.echo_tail_ms / 1000.0
        while not self._stop.is_set():
            job = self._tts_q.get()
            if job is None:
                break
            if job.turn != self._turn_id:
                continue
            try:
                if job.end:
                    while time.monotonic() < self._play_end and job.turn == self._turn_id and not self._stop.is_set():
                        time.sleep(0.03)
                    if job.turn == self._turn_id:
                        time.sleep(tail)
                        if job.turn == self._turn_id:
                            self._set_state(State.IDLE)
                            self.emit("speaking_done")
                    continue
                text = clean_for_tts(job.text)
                if not has_speakable(text):
                    continue
                t = time.perf_counter()
                audio = self.tts.synth(text)
                tts_ms = (time.perf_counter() - t) * 1000
                if job.turn != self._turn_id:
                    continue
                self._emit_audio(job, audio, tts_ms)
            except Exception as e:                  # noqa: BLE001
                log.exception("tts failed")
                self.emit("error", where="tts", error=e)

    def _emit_audio(self, job: _TTSJob, audio: np.ndarray, tts_ms: float) -> None:
        sr = self.tts.sample_rate
        audio = _fade(audio.astype(np.float32, copy=False), sr)
        now = time.monotonic()
        with self._turn_lock:
            if job.turn != self._turn_id:
                return
            start = max(now, self._play_end)
            end = start + len(audio) / sr
            self._play_end = end
            self._chunks.append(_Chunk(job.turn, job.text, start, end, job.filler))
        first = len([c for c in self._chunks if not c.filler]) == 1 and not job.filler
        if self._playback is not None:
            self._playback.write(audio, sr)
        self.emit("audio_out", audio=audio, sr=sr)
        self._set_state(State.SPEAKING)
        m = self._turn_metrics
        if (first or job.filler) and "first_audio_ms" not in m and m.get("t_speech_end"):
            m["tts_first_ms"] = tts_ms
            m["first_audio_ms"] = (now - m["t_speech_end"]) * 1000
            self.metrics_log.append(dict(m))
            self.emit("metrics", **m)

    # ================================================================= interruption
    def _spoken_text(self, turn: int, now: float) -> str:
        words: List[str] = []
        for ch in self._chunks:
            if ch.turn != turn or ch.filler or ch.start >= now:
                continue
            w = ch.text.split()
            if ch.end <= now:
                words += w
            else:
                frac = (now - ch.start) / max(1e-3, ch.end - ch.start)
                words += w[: int(len(w) * frac)]
        return " ".join(words)

    def _interrupt(self, reason: str) -> None:
        with self._turn_lock:
            if self._state not in (State.THINKING, State.SPEAKING):
                return
            now = time.monotonic()
            turn = self._turn_id
            spoken = self._spoken_text(turn, now)
            was_speaking = self._state == State.SPEAKING
            self._cancel.set()
            self._turn_id += 1
            self._cancel_info[turn] = spoken
            while True:                              # drop queued synthesis work
                try:
                    self._tts_q.get_nowait()
                except queue.Empty:
                    break
            self._play_end = 0.0
            self._chunks = []
            if self._playback is not None:
                self._playback.flush()
        self.emit("audio_flush")
        with self._hist_lock:
            msg = self._assistant_msgs.pop(turn, None)
            if msg is not None and msg in self.history:      # reply already committed to history
                if spoken:
                    msg.content = spoken
                else:
                    self.history.remove(msg)
                self._cancel_info.pop(turn, None)
            elif not was_speaking and reason in ("user_speech", "barge_in") and self.history \
                    and self.history[-1].role == "user" and not spoken:
                # user started talking again before we said anything: merge their thoughts
                last = self.history.pop()
                self._carry_text = last.content.split("\n[voice cues:")[0]
        self.emit("interrupted", reason=reason, spoken=spoken)
        self._set_state(State.LISTENING if reason in ("user_speech", "barge_in") else State.IDLE)


# ================================================================= helpers
def _dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def _fade(a: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
    n = min(len(a) // 2, int(sr * ms / 1000))
    if n <= 1:
        return a
    a = a.copy()
    r = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a[:n] *= r
    a[-n:] *= r[::-1]
    return a
