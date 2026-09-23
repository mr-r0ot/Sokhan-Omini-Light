"""Google Colab (and Jupyter-in-Colab) bridge: talk to Sokhan from the browser.

In a notebook the microphone, the speakers and the webcam live in the
*browser*, not in Python. ``ColabBridge`` renders a small voice UI in the
cell output and moves audio through one bidirectional channel: every ~60 ms
the page sends the microphone audio it captured and receives the reply audio
and UI events in the same round trip (``google.colab.kernel.invokeFunction``).
No servers, no ports, no ``eval_js`` from background threads.

    from sokhan.colab import ColabBridge
    ColabBridge(omni).show()

Keep the kernel idle afterwards (don't run a blocking cell): the round trips
are served between cell executions. The browser's echo cancellation keeps the
assistant from hearing itself, so barge-in works on laptop speakers too.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .audio import resample, to_pcm16

log = logging.getLogger("sokhan.colab")

_UI_EVENTS = {"state", "partial_transcript", "transcript", "response_delta", "response_done", "interrupted",
              "metrics", "tool_call", "warning", "error", "level"}


class ColabBridge:
    def __init__(self, omni, title: str = "Sokhan", camera: bool = False, camera_every_s: float = 3.0):
        self.omni = omni
        self.title = title
        self.camera = camera and bool(omni.capabilities.get("vision"))
        self.camera_every_s = camera_every_s
        self._audio: List[np.ndarray] = []
        self._events: List[Dict[str, Any]] = []
        self._flush = False
        self._browser_queue = 0.0
        self._lock = threading.Lock()
        self._unsub = None
        self._level_t = 0.0

    # ------------------------------------------------------------------ sink protocol
    def write(self, audio: np.ndarray, sr: int) -> None:
        a = resample(np.asarray(audio, np.float32), sr, 24000)
        with self._lock:
            self._audio.append(a)

    def flush(self) -> None:
        with self._lock:
            self._audio.clear()
            self._flush = True

    def pending_seconds(self) -> float:
        with self._lock:
            unsent = sum(len(a) for a in self._audio) / 24000
        return unsent + self._browser_queue

    def level_db(self) -> Optional[float]:
        return None                      # the browser cancels the echo for us

    # ------------------------------------------------------------------ events -> UI
    def _on_event(self, event: str, **kw) -> None:
        if event not in _UI_EVENTS:
            return
        if event == "level":
            now = time.monotonic()
            if now - self._level_t < 0.1:
                return
            self._level_t = now
        data: Dict[str, Any] = {"e": event}
        for k, v in kw.items():
            if k == "state":
                v = getattr(v, "value", str(v))
            elif k == "error":
                v = f"{type(v).__name__}: {v}"
            elif k == "features":
                continue
            data[k] = v if isinstance(v, (str, int, float, bool, type(None), dict, list)) else str(v)
        with self._lock:
            self._events.append(data)
            if len(self._events) > 400:
                del self._events[:100]

    # ------------------------------------------------------------------ browser round trip
    def _tick(self, mic_b64: str = "", queued: float = 0.0, cmd: str = "") -> Any:
        from IPython.display import JSON  # type: ignore
        self._browser_queue = float(queued or 0.0)
        if mic_b64:
            pcm = np.frombuffer(base64.b64decode(mic_b64), dtype=np.int16)
            self.omni.feed_audio(pcm, 16000)
        if cmd:
            self._command(json.loads(cmd))
        with self._lock:
            audio, self._audio = self._audio, []
            events, self._events = self._events, []
            flush, self._flush = self._flush, False
        out: Dict[str, Any] = {"ev": events, "flush": flush}
        if audio:
            out["audio"] = base64.b64encode(to_pcm16(np.concatenate(audio)).tobytes()).decode("ascii")
        return JSON(out)

    def _command(self, c: Dict[str, Any]) -> None:
        kind = c.get("type")
        if kind == "text" and c.get("text"):
            self.omni.send_text(c["text"])
        elif kind == "interrupt":
            self.omni.interrupt()
        elif kind == "reset":
            self.omni.reset()
        elif kind == "image" and c.get("data"):
            self.omni.set_image(c["data"])

    # ------------------------------------------------------------------ display
    def show(self) -> "ColabBridge":
        try:
            from google.colab import output  # type: ignore
        except ImportError as e:
            raise RuntimeError("ColabBridge needs Google Colab (google.colab is not importable)") from e
        from IPython.display import HTML, display  # type: ignore
        self.omni.wait_ready()
        output.register_callback("sokhan.tick", self._tick)
        self.omni.attach_sink(self)
        if self._unsub is None:
            self._unsub = self.omni.on("*", self._on_event)
        html = _UI.replace("__TITLE__", self.title).replace("__CAM__", "true" if self.camera else "false") \
                  .replace("__CAM_EVERY__", str(int(self.camera_every_s * 1000)))
        display(HTML(html))
        return self

    def close(self) -> None:
        self.omni.detach_sink(self)
        if self._unsub:
            self._unsub()
            self._unsub = None


_UI = r"""
<div id="sk-root">
<style>
#sk-root{--bg:#0f1117;--card:#171a23;--line:#262b38;--ink:#e8eaf0;--mute:#8a90a2;--acc:#7c8cff;--acc2:#35d0ba;--warn:#ffb454;
 font-family:Inter,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg);border-radius:18px;padding:22px;max-width:760px}
#sk-root .top{display:flex;align-items:center;gap:18px}
#sk-root .orb{width:74px;height:74px;border-radius:50%;flex:none;background:radial-gradient(circle at 35% 30%,#fff5,transparent 45%),linear-gradient(135deg,var(--acc),var(--acc2));
 box-shadow:0 0 0 0 #7c8cff55;transition:transform .12s ease,filter .3s}
#sk-root .orb.idle{filter:saturate(.35) brightness(.8)}
#sk-root .orb.listening{animation:skp 1.4s infinite}
#sk-root .orb.thinking{animation:skspin 1.1s linear infinite;background:conic-gradient(var(--acc),var(--acc2),var(--acc))}
#sk-root .orb.speaking{animation:skp .7s infinite;filter:brightness(1.15)}
@keyframes skp{0%{box-shadow:0 0 0 0 #7c8cff66}100%{box-shadow:0 0 0 22px #7c8cff00}}
@keyframes skspin{to{transform:rotate(360deg)}}
#sk-root h2{margin:0;font-size:20px;font-weight:650;letter-spacing:.2px}
#sk-root .st{color:var(--mute);font-size:13px;margin-top:4px}
#sk-root .bar{height:4px;background:var(--line);border-radius:4px;margin-top:10px;overflow:hidden;width:220px}
#sk-root .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc2),var(--acc));transition:width .1s}
#sk-root .log{margin-top:18px;height:330px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding-right:6px}
#sk-root .m{max-width:82%;padding:10px 14px;border-radius:14px;line-height:1.65;font-size:15px;white-space:pre-wrap;unicode-bidi:plaintext;text-align:start}
#sk-root .u{align-self:flex-end;background:#2a3050;border-bottom-right-radius:4px}
#sk-root .a{align-self:flex-start;background:var(--card);border:1px solid var(--line);border-bottom-left-radius:4px}
#sk-root .p{align-self:flex-end;color:var(--mute);font-style:italic;background:transparent;border:1px dashed var(--line)}
#sk-root .s{align-self:center;color:var(--mute);font-size:12px}
#sk-root .row{display:flex;gap:8px;margin-top:14px}
#sk-root input{flex:1;background:var(--card);border:1px solid var(--line);color:var(--ink);border-radius:12px;padding:11px 14px;font-size:14px;outline:none;unicode-bidi:plaintext}
#sk-root button{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:12px;padding:10px 14px;cursor:pointer;font-size:14px}
#sk-root button.pri{background:linear-gradient(135deg,var(--acc),#5b6cf0);border:none;font-weight:600}
#sk-root button:hover{filter:brightness(1.15)}
#sk-root .met{color:var(--mute);font-size:12px;margin-top:10px;min-height:16px}
</style>
<div class="top"><div class="orb idle" id="sk-orb"></div>
 <div><h2>__TITLE__</h2><div class="st" id="sk-st">Press “Start talking” and allow the microphone</div>
 <div class="bar"><i id="sk-lvl"></i></div></div></div>
<div class="log" id="sk-log"></div>
<div class="row"><input id="sk-in" placeholder="…or type a message and press Enter">
 <button class="pri" id="sk-mic">🎙 Start talking</button><button id="sk-stop">■ Stop</button><button id="sk-cam" style="display:none">📷 Camera</button></div>
<div class="met" id="sk-met"></div>
</div>
<script>
(() => {
 const $ = id => document.getElementById(id);
 const log = $('sk-log'), orb = $('sk-orb'), st = $('sk-st');
 const labels = {idle:'Ready — just talk', listening:'Listening…', thinking:'Thinking…', speaking:'Speaking — talk to interrupt', loading:'Loading…'};
 let running = false, micBuf = [], outCtx = null, playHead = 0, nodes = [], cur = null, partial = null, cmds = [];
 let camOn = false, camStream = null, camVideo = null, lastCam = 0;
 const CAM = __CAM__, CAM_EVERY = __CAM_EVERY__;
 if (CAM) $('sk-cam').style.display = '';
 const add = (cls, text) => { const d = document.createElement('div'); d.className = 'm ' + cls; d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; };
 const b64 = u8 => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
 const unb64 = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };

 function play(pcm16) {
   if (!outCtx) { outCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000}); playHead = outCtx.currentTime; }
   const f = new Float32Array(pcm16.length); for (let i = 0; i < f.length; i++) f[i] = pcm16[i] / 32768;
   const buf = outCtx.createBuffer(1, f.length, 24000); buf.copyToChannel(f, 0);
   const n = outCtx.createBufferSource(); n.buffer = buf; n.connect(outCtx.destination);
   const t = Math.max(outCtx.currentTime + 0.03, playHead); n.start(t); playHead = t + buf.duration;
   nodes.push(n); n.onended = () => { nodes = nodes.filter(x => x !== n); };
 }
 function flush() { nodes.forEach(n => { try { n.stop(); } catch (e) {} }); nodes = []; if (outCtx) playHead = outCtx.currentTime; }
 const queued = () => outCtx ? Math.max(0, playHead - outCtx.currentTime) : 0;

 function onEvent(e) {
   switch (e.e) {
     case 'state': orb.className = 'orb ' + e.state; st.textContent = labels[e.state] || e.state; break;
     case 'level': $('sk-lvl').style.width = Math.min(100, e.speech * 100) + '%'; break;
     case 'partial_transcript': if (!partial) partial = add('p', ''); partial.textContent = e.text; break;
     case 'transcript': if (partial) { partial.remove(); partial = null; } add('u', e.text); cur = null; break;
     case 'response_delta': if (!cur) cur = add('a', ''); cur.textContent += e.text; log.scrollTop = log.scrollHeight; break;
     case 'response_done': cur = null; break;
     case 'interrupted': if (partial) { partial.remove(); partial = null; } add('s', '⤺ interrupted'); cur = null; break;
     case 'tool_call': add('s', '🔧 ' + e.name + ' ' + JSON.stringify(e.arguments)); break;
     case 'metrics': $('sk-met').textContent = `latency ${Math.round(e.latency_ms)} ms · stt ${Math.round(e.stt_ms||0)} · llm first token ${Math.round(e.llm_ttft_ms||0)} · first voice ${Math.round(e.tts_first_ms||0)} ms`; break;
     case 'warning': add('s', '⚠ ' + e.message); break;
     case 'error': add('s', '✗ ' + e.stage + ': ' + e.error); break;
   }
 }

 async function grabCam() {
   if (!camOn || !camVideo || Date.now() - lastCam < CAM_EVERY) return;
   lastCam = Date.now();
   const c = document.createElement('canvas'); c.width = 448; c.height = Math.round(448 * camVideo.videoHeight / Math.max(1, camVideo.videoWidth));
   c.getContext('2d').drawImage(camVideo, 0, 0, c.width, c.height);
   cmds.push(JSON.stringify({type: 'image', data: c.toDataURL('image/jpeg', 0.75)}));
 }

 async function loop() {
   while (running) {
     let mic = '';
     if (micBuf.length) {
       let n = 0; micBuf.forEach(a => n += a.length);
       const all = new Int16Array(n); let o = 0; micBuf.forEach(a => { all.set(a, o); o += a.length; }); micBuf = [];
       mic = b64(new Uint8Array(all.buffer));
     }
     await grabCam();
     const cmd = cmds.length ? cmds.shift() : '';
     try {
       const r = await google.colab.kernel.invokeFunction('sokhan.tick', [mic, queued(), cmd], {});
       const d = r.data['application/json'];
       if (d.flush) flush();
       if (d.audio) { const u = unb64(d.audio); play(new Int16Array(u.buffer)); }
       d.ev.forEach(onEvent);
     } catch (err) { st.textContent = 'Connection lost — re-run the cell'; running = false; break; }
     await new Promise(r => setTimeout(r, 40));
   }
 }

 async function startMic() {
   const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}});
   const ctx = new (window.AudioContext || window.webkitAudioContext)();
   const src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(2048, 1, 1);
   const ratio = ctx.sampleRate / 16000;
   proc.onaudioprocess = ev => {
     const x = ev.inputBuffer.getChannelData(0), n = Math.floor(x.length / ratio), y = new Int16Array(n);
     for (let i = 0; i < n; i++) { const v = x[Math.floor(i * ratio)]; y[i] = Math.max(-1, Math.min(1, v)) * 32767; }
     micBuf.push(y);
   };
   src.connect(proc); proc.connect(ctx.destination);
   if (!outCtx) { outCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000}); playHead = outCtx.currentTime; }
 }

 $('sk-mic').onclick = async () => {
   if (running) return;
   try { await startMic(); } catch (e) { st.textContent = 'Microphone blocked: ' + e.message + ' (you can still type)'; }
   running = true; $('sk-mic').textContent = '● Live'; loop();
 };
 $('sk-stop').onclick = () => { flush(); cmds.push(JSON.stringify({type: 'interrupt'})); };
 $('sk-in').onkeydown = e => {
   if (e.key !== 'Enter' || !e.target.value.trim()) return;
   cmds.push(JSON.stringify({type: 'text', text: e.target.value.trim()})); e.target.value = '';
   if (!running) { running = true; if (!outCtx) { outCtx = new AudioContext({sampleRate: 24000}); playHead = outCtx.currentTime; } loop(); }
 };
 $('sk-cam').onclick = async () => {
   camOn = !camOn; $('sk-cam').textContent = camOn ? '📷 On' : '📷 Camera';
   if (camOn) { camStream = await navigator.mediaDevices.getUserMedia({video: {width: 640}}); camVideo = document.createElement('video'); camVideo.srcObject = camStream; camVideo.muted = true; await camVideo.play(); }
   else if (camStream) { camStream.getTracks().forEach(t => t.stop()); camVideo = null; }
 };
})();
</script>
"""
