"""LLM backends.

``llama_cpp`` (default)  in-process llama.cpp, any GGUF model, CPU or GPU
``openai``               any OpenAI-compatible server (Ollama, vLLM, LM Studio, cloud APIs)
``llama_server``         a managed llama.cpp ``llama-server`` subprocess
``mock``                 scripted replies for tests

Latency engineering in the in-process backend
---------------------------------------------
On a CPU, prompt prefill is the slowest thing in the whole pipeline (tens of
tokens per second), so the backend never re-reads what it has already read:

* The conversation is kept **token-exact and append-only**: every message is
  tokenized once, and a reply is stored as the exact token ids the model
  generated. Each new turn therefore only prefills the new user message.
* When history *does* change (a speculative turn was dropped, old turns were
  compacted away) the cache is rolled back: partially for attention models,
  or by restoring a saved **checkpoint** for hybrid/recurrent models such as
  Qwen3.5, whose state cannot be truncated.
"""
from __future__ import annotations

import codecs
import json
import logging
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .registry import LoadContext, register

log = logging.getLogger("sokhan.llm")

Piece = Union[str, Dict[str, Any]]        # text, or a native tool call {"name", "arguments"}


@dataclass
class ChatMessage:
    role: str                              # system | user | assistant | tool
    content: str
    images: List[str] = field(default_factory=list)        # data URLs
    meta: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


class ContextFull(RuntimeError):
    pass


# --------------------------------------------------------------------------- tool prompt
def tools_prompt(tools: Sequence[Dict[str, Any]], xml: bool) -> str:
    if not tools:
        return ""
    specs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    if xml:                                # Qwen3.5 native wording
        call = ("<tool_call>\n<function=function_name>\n<parameter=parameter_name>\nvalue\n</parameter>\n"
                "</function>\n</tool_call>")
    else:
        call = '<tool_call>\n{"name": "function_name", "arguments": {"parameter_name": "value"}}\n</tool_call>'
    return ("\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + specs + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" + call +
            "\n\nThe result comes back in a <tool_response>. Then answer the user in plain speech; "
            "never read tool calls or raw results aloud.")


class LLMBackend:
    """Implement ``stream``; the rest is optional."""
    supports_images = False

    def __init__(self, config=None):
        self.config = config

    def load(self, ctx: LoadContext) -> None: ...

    def stream(self, messages: List[ChatMessage], cancel: threading.Event,
               tools: Sequence[Dict[str, Any]] = ()) -> Iterator[Piece]:
        raise NotImplementedError

    def prefill(self, messages: List[ChatMessage], tools: Sequence[Dict[str, Any]] = (),
                cancel: Optional[threading.Event] = None) -> None:
        """Pre-read a prompt (system prompt at startup, compacted history) - optional."""

    def bind_reply(self, message: ChatMessage) -> None:
        """Attach backend state (exact token ids) of the last generation to its history message."""

    def checkpoint(self) -> None:
        """Snapshot the model state at a turn boundary (idle time) - optional."""

    def count_tokens(self, message: ChatMessage) -> int:
        return len(message.content) // 3 + 8

    def close(self) -> None: ...


# =========================================================================== llama.cpp
@dataclass
class _Checkpoint:
    tokens: Tuple[int, ...]
    state: Any
    pinned: bool = False


@register("llm", "llama_cpp", "llama.cpp", "llamacpp")
class LlamaCppLLM(LLMBackend):
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self._lock = threading.RLock()
        self._key = "tok:" + uuid.uuid4().hex[:8]
        self._ckpts: List[_Checkpoint] = []
        self._last: Optional[Tuple[List[int], List[int]]] = None       # (header ids, generated ids)
        self.stateful = False                                           # recurrent/hybrid memory
        self.chatml = True
        self.xml_tools = False
        self.think_tags = False

    # ------------------------------------------------------------------ load
    def load(self, ctx: LoadContext) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise RuntimeError("the default LLM needs `pip install llama-cpp-python` "
                               "(or set llm.backend = 'openai')") from e
        from . import models
        cfg, l, plan = self.config, self.config.llm, ctx.plan
        path, mmproj = models.resolve_llm(cfg, ctx.progress, need_mmproj=cfg.vision.enabled)
        ctx.report("llm", 0.95, "loading language model")
        kw: Dict[str, Any] = dict(
            model_path=path, n_ctx=l.n_ctx, n_batch=l.n_batch, n_threads=plan.llm_threads,
            n_threads_batch=plan.llm_batch_threads, n_gpu_layers=(plan.llm_gpu_layers if plan.use_gpu else 0),
            use_mmap=l.use_mmap, use_mlock=l.use_mlock, flash_attn=l.flash_attn, verbose=False)
        if l.seed >= 0:
            kw["seed"] = l.seed
        if mmproj:
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler  # type: ignore
                kw["chat_handler"] = MTMDChatHandler(clip_model_path=mmproj, verbose=False, use_gpu=plan.use_gpu)
                self.supports_images = True
            except Exception as e:
                log.warning("vision projector unavailable (%s); images will be ignored", e)
        elif cfg.vision.enabled:
            log.warning("vision enabled but no mmproj file was found for %s", l.model)
        kw.update(l.extra)
        while True:                        # tolerate older builds that lack a keyword
            try:
                self.llm = Llama(**kw)
                break
            except TypeError as e:
                bad = next((k for k in list(kw) if f"'{k}'" in str(e)), None)
                if not bad or bad == "model_path":
                    raise
                kw.pop(bad)
        import llama_cpp as lc  # type: ignore
        mdl = getattr(self.llm._model, "model", None)
        for fn in ("llama_model_is_recurrent", "llama_model_is_hybrid"):
            try:
                self.stateful = self.stateful or bool(getattr(lc, fn)(mdl))
            except Exception:
                pass
        self._vocab = getattr(self.llm._model, "vocab", None)
        self._eos = self.llm.token_eos()
        tpl = str(self.llm.metadata.get("tokenizer.chat_template", ""))
        fmt = l.chat_format.lower()
        self.chatml = fmt == "chatml" or (fmt == "auto" and ("<|im_start|>" in tpl or not tpl))
        self.xml_tools = "<function=" in tpl
        self.think_tags = "<think>" in tpl
        self._jinja = None
        if tpl:
            try:
                from llama_cpp.llama_chat_format import Jinja2ChatFormatter  # type: ignore
                eos = self.llm.detokenize([self._eos], special=True).decode("utf-8", "ignore")
                bos_id = self.llm.token_bos()
                bos = self.llm.detokenize([bos_id], special=True).decode("utf-8", "ignore") if bos_id >= 0 else ""
                self._jinja = Jinja2ChatFormatter(template=tpl, eos_token=eos, bos_token=bos)
            except Exception as e:
                log.debug("chat template unusable (%s); using built-in ChatML", e)
                self.chatml = True
        self._end_ids = self._tok("<|im_end|>\n") if self.chatml else []

    # ------------------------------------------------------------------ rendering
    def _tok(self, text: str) -> List[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def _header(self) -> str:
        if self.config.llm.thinking and self.think_tags:
            return "<|im_start|>assistant\n<think>\n"
        return "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if self.think_tags else "")

    def _system_block(self, content: str, tools: Sequence[Dict[str, Any]]) -> str:
        """System turn exactly as the model was trained to see it (tools included), rendered
        with the GGUF's own chat template; our generic wording is only the fallback."""
        if tools and self._jinja is not None:
            key = (content, json.dumps(list(tools), sort_keys=True))
            if getattr(self, "_sys_cache", (None, ""))[0] == key:
                return self._sys_cache[1]
            try:
                p = self._jinja(messages=[{"role": "system", "content": content},
                                          {"role": "user", "content": "\x01"}], tools=list(tools)).prompt
                i = p.find("<|im_start|>user")
                if i > 0 and "<tools>" in p[:i]:
                    self._sys_cache = (key, p[:i])
                    return p[:i]
            except Exception as e:
                log.debug("template tools rendering failed: %s", e)
        return f"<|im_start|>system\n{content}{tools_prompt(tools, self.xml_tools)}<|im_end|>\n"

    def _msg_ids(self, m: ChatMessage, tools: Sequence[Dict[str, Any]]) -> List[int]:
        ids = m.meta.get(self._key)
        if ids is not None:
            return ids
        if m.role == "system":
            text = self._system_block(m.content, tools)
        elif m.role == "tool":
            text = f"<|im_start|>user\n<tool_response>\n{m.content}\n</tool_response><|im_end|>\n"
        elif m.role == "assistant":        # (generated replies carry their exact ids already)
            text = ("<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if self.think_tags else "")
                    + m.content + "<|im_end|>\n")
        else:
            text = f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
        ids = self._tok(text)
        if m.role != "system":             # the system block depends on the tool list
            m.meta[self._key] = ids
        return ids

    def _prompt(self, messages: List[ChatMessage], tools: Sequence[Dict[str, Any]],
                generation: bool = True) -> List[int]:
        if self.chatml:
            ids: List[int] = []
            for m in messages:
                ids += self._msg_ids(m, tools)
            return ids + (self._tok(self._header()) if generation else [])
        msgs = [{"role": "user" if m.role == "tool" else m.role, "content": m.content} for m in messages]
        resp = self._jinja(messages=msgs, tools=list(tools) or None)
        text = resp.prompt
        if not generation:                 # cut the generation prompt off
            text = text[: text.rfind(msgs[-1]["content"]) + len(msgs[-1]["content"])] if msgs else text
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=not resp.added_special, special=True))

    def count_tokens(self, message: ChatMessage) -> int:
        if self.llm is None or not self.chatml:
            return super().count_tokens(message)
        return len(self._msg_ids(message, ()))

    # ------------------------------------------------------------------ cache management
    def _prepare(self, tokens: List[int], need_logits: bool) -> None:
        """Make the model state a prefix of ``tokens`` as cheaply as possible."""
        llm = self.llm
        n = llm.n_tokens
        limit = len(tokens) - 1 if need_logits else len(tokens)
        cur = llm._input_ids
        lcp = 0
        for a, b in zip(cur, tokens):
            if a != b:
                break
            lcp += 1
        if lcp == n and n <= limit:
            return                                       # pure append: the common case
        target = min(lcp, limit)
        if target > 0 and llm._ctx.kv_cache_seq_rm(-1, target, -1):
            llm.n_tokens = target                        # attention models: cut in place
            return
        best: Optional[_Checkpoint] = None
        for cp in self._ckpts:
            L = len(cp.tokens)
            if L <= limit and (best is None or L > len(best.tokens)) and tuple(tokens[:L]) == cp.tokens:
                best = cp
        if best is not None:
            llm.load_state(best.state)
            log.debug("restored checkpoint at %d tokens", len(best.tokens))
        else:
            log.debug("full prompt re-read (%d tokens)", len(tokens))
            llm.reset()

    def checkpoint(self, pinned: bool = False) -> None:
        if not self.stateful or self.llm is None or self.config.llm.checkpoints <= 0:
            return
        with self._lock:
            n = self.llm.n_tokens
            if n == 0:
                return
            toks = tuple(int(t) for t in self.llm._input_ids)
            if any(cp.tokens == toks for cp in self._ckpts):
                return
            self._ckpts.append(_Checkpoint(toks, self.llm.save_state(), pinned))
            keep = max(1, self.config.llm.checkpoints)
            while len(self._ckpts) > keep:
                victim = next((c for c in self._ckpts if not c.pinned), None)
                if victim is None:
                    break
                self._ckpts.remove(victim)

    def prefill(self, messages: List[ChatMessage], tools: Sequence[Dict[str, Any]] = (),
                cancel: Optional[threading.Event] = None) -> None:
        """Read a prompt ahead of time. Interruptible: a partial read is still a valid prefix."""
        with self._lock:
            toks = self._prompt(messages, tools, generation=False)
            self._prepare(toks, need_logits=False)
            step = max(16, self.config.llm.n_batch)
            while self.llm.n_tokens < len(toks):
                if cancel is not None and cancel.is_set():
                    return
                n = self.llm.n_tokens
                self.llm.eval(toks[n:n + step])
        self.checkpoint(pinned=len(messages) == 1)

    # ------------------------------------------------------------------ generation
    def _is_eog(self, tok: int) -> bool:
        if tok == self._eos:
            return True
        try:
            import llama_cpp as lc  # type: ignore
            return bool(lc.llama_vocab_is_eog(self._vocab, tok))
        except Exception:
            return False

    def stream(self, messages, cancel, tools=()):
        if self.supports_images and any(m.images for m in messages):
            yield from self._stream_vision(messages, cancel, tools)
            return
        l = self.config.llm
        with self._lock:
            tokens = self._prompt(messages, tools)
            room = l.n_ctx - len(tokens) - 1
            if room < 16:
                raise ContextFull(f"prompt ({len(tokens)} tokens) does not fit n_ctx={l.n_ctx}")
            self._prepare(tokens, need_logits=True)
            header = self._tok(self._header()) if self.chatml else []
            gen: List[int] = []
            self._last = (header, gen)
            if l.seed >= 0:
                self.llm.set_seed(l.seed)
            dec = codecs.getincrementaldecoder("utf-8")(errors="ignore")
            it = self.llm.generate(tokens, top_k=l.top_k, top_p=l.top_p, min_p=l.min_p, temp=l.temperature,
                                   repeat_penalty=l.repeat_penalty, presence_penalty=l.presence_penalty,
                                   frequency_penalty=l.frequency_penalty, reset=True)
            tail = ""
            try:
                for tok in it:
                    if cancel.is_set() or self._is_eog(tok):
                        break
                    gen.append(int(tok))
                    piece = dec.decode(self.llm.detokenize([tok], special=True))
                    if piece:
                        tail = (tail + piece)[-24:]
                        if "<|im_end|>" in tail or "<|im_start|>" in tail or "<|endoftext|>" in tail:
                            break
                        yield piece
                    if len(gen) >= min(l.max_tokens, room):
                        break
            finally:
                it.close()

    def bind_reply(self, message: ChatMessage) -> None:
        if self._last is None or not self.chatml:
            return
        header, gen = self._last
        message.meta[self._key] = header + list(gen) + self._end_ids
        self._last = None

    def _stream_vision(self, messages, cancel, tools):
        l = self.config.llm
        self._last = None
        msgs = []
        for i, m in enumerate(messages):
            content: Any = m.content
            if i == 0 and m.role == "system":
                content = m.content + tools_prompt(tools, self.xml_tools)
            if m.images:
                content = [{"type": "image_url", "image_url": {"url": u}} for u in m.images] + \
                          [{"type": "text", "text": m.content}]
            msgs.append({"role": "user" if m.role == "tool" else m.role, "content": content})
        with self._lock:
            it = self.llm.create_chat_completion(messages=msgs, stream=True, max_tokens=l.max_tokens,
                                                 temperature=l.temperature, top_p=l.top_p, top_k=l.top_k,
                                                 min_p=l.min_p, repeat_penalty=l.repeat_penalty)
            try:
                for ch in it:
                    if cancel.is_set():
                        break
                    piece = ch["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece
            finally:
                close = getattr(it, "close", None)
                if close:
                    close()

    def close(self) -> None:
        with self._lock:
            self._ckpts.clear()
            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
            self.llm = None


# =========================================================================== OpenAI-compatible
@register("llm", "openai", "openai_compat")
class OpenAILLM(LLMBackend):
    """Any /v1/chat/completions endpoint with streaming (SSE)."""
    supports_images = True

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.llm.base_url.rstrip("/")

    def load(self, ctx: LoadContext) -> None: ...

    def _payload(self, messages, tools, max_tokens=None) -> Dict[str, Any]:
        l = self.config.llm
        out = []
        for i, m in enumerate(messages):
            text = m.content + (tools_prompt(tools, False) if i == 0 and m.role == "system" else "")
            if m.role == "tool":
                out.append({"role": "user", "content": f"<tool_response>\n{m.content}\n</tool_response>"})
            elif m.images:
                out.append({"role": m.role, "content": [{"type": "image_url", "image_url": {"url": u}}
                                                        for u in m.images] + [{"type": "text", "text": text}]})
            else:
                out.append({"role": m.role, "content": text})
        body: Dict[str, Any] = {"messages": out, "stream": True, "max_tokens": max_tokens or l.max_tokens,
                                "temperature": l.temperature, "top_p": l.top_p}
        if l.model_name:
            body["model"] = l.model_name
        if l.seed >= 0:
            body["seed"] = l.seed
        if l.presence_penalty:
            body["presence_penalty"] = l.presence_penalty
        if l.frequency_penalty:
            body["frequency_penalty"] = l.frequency_penalty
        if self.config.llm.backend in ("llama_server",):  # llama.cpp understands these extras
            body.update({"top_k": l.top_k, "min_p": l.min_p, "repeat_penalty": l.repeat_penalty,
                         "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": l.thinking}})
        body.update(l.extra)
        return body

    def stream(self, messages, cancel, tools=()):
        l = self.config.llm
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if l.api_key:
            headers["Authorization"] = f"Bearer {l.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(self._payload(messages, tools)).encode(), headers=headers)
        resp = urllib.request.urlopen(req, timeout=l.timeout_s)
        try:
            for raw in resp:
                if cancel.is_set():
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except Exception:
                    continue
                if delta.get("content"):
                    yield delta["content"]
        finally:
            resp.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@register("llm", "llama_server")
class LlamaServerLLM(OpenAILLM):
    """Spawns llama.cpp's ``llama-server``: newest architectures, GPU builds for every vendor."""

    def __init__(self, config):
        super().__init__(config)
        self.proc: Optional[subprocess.Popen] = None

    def load(self, ctx: LoadContext) -> None:
        from . import models
        l, plan = self.config.llm, ctx.plan
        binary = l.server_binary or shutil.which("llama-server") or shutil.which("llama-server.exe")
        if not binary:
            raise RuntimeError("llama-server not found: install llama.cpp or set llm.server_binary")
        path, mmproj = models.resolve_llm(self.config, ctx.progress, need_mmproj=self.config.vision.enabled)
        port = _free_port()
        cmd = [binary, "-m", path, "-c", str(l.n_ctx), "-b", str(l.n_batch), "--host", "127.0.0.1",
               "--port", str(port), "--jinja", "-t", str(plan.llm_threads)]
        if plan.use_gpu:
            cmd += ["-ngl", "999" if plan.llm_gpu_layers < 0 else str(plan.llm_gpu_layers)]
        if mmproj:
            cmd += ["--mmproj", mmproj]
        cmd += list(l.server_args)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base_url = f"http://127.0.0.1:{port}/v1"
        deadline = time.time() + max(90.0, l.timeout_s)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited (code {self.proc.returncode}); run it by hand to "
                                   f"see why: {' '.join(cmd)}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.3)
        raise RuntimeError("llama-server did not become ready in time")

    def prefill(self, messages, tools=(), cancel=None):
        try:                                             # a 1-token request primes the prompt cache
            ev = threading.Event()
            old, self.config.llm.max_tokens = self.config.llm.max_tokens, 1
            try:
                for _ in self.stream(messages + [ChatMessage("user", ".")], ev, tools):
                    pass
            finally:
                self.config.llm.max_tokens = old
        except Exception as e:
            log.debug("server warmup failed: %s", e)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# =========================================================================== mock
@register("llm", "mock")
class MockLLM(LLMBackend):
    """Scripted LLM for tests / UI work. ``replies`` items may be callables(messages) -> str."""
    supports_images = True

    def __init__(self, config=None, replies=None, token_delay: float = 0.004, first_delay: float = 0.05):
        super().__init__(config)
        self.replies, self.i = list(replies or ["Hello! How can I help you?"]), 0
        self.token_delay, self.first_delay = token_delay, first_delay
        self.seen: List[List[ChatMessage]] = []

    def stream(self, messages, cancel, tools=()):
        self.seen.append(list(messages))
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        text = r(messages) if callable(r) else r
        time.sleep(self.first_delay)
        for k in range(0, len(text), 3):
            if cancel.is_set():
                return
            time.sleep(self.token_delay)
            yield text[k:k + 3]
