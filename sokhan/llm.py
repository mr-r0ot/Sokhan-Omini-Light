"""LLM backends: in-process llama.cpp (default), a managed ``llama-server``
subprocess (needed for image input), or any OpenAI-compatible endpoint.

All backends expose ``stream(messages, cancel) -> Iterator[str]`` and speak
the same Qwen ChatML/tool-call text protocol, so swapping them is one config line.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from . import models
from .config import Config

log = logging.getLogger("sokhan.llm")

EMPTY_THINK = "<think>\n\n</think>\n\n"


@dataclass
class ChatMessage:
    role: str
    content: str
    images: List[str] = field(default_factory=list)     # data URLs


def render_chatml(messages: List[ChatMessage], *, add_generation_prompt: bool = True,
                  enable_thinking: bool = False, keep_empty_think: bool = True) -> str:
    """Qwen ChatML.  History replies get the same empty <think> block the model
    produced, so the token prefix stays byte-identical -> llama.cpp reuses its KV cache."""
    out = []
    for m in messages:
        c = m.content
        if m.role == "assistant" and keep_empty_think and not enable_thinking:
            c = EMPTY_THINK + c
        out.append(f"<|im_start|>{m.role}\n{c}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else EMPTY_THINK))
    return "".join(out)


class LLMBackend(ABC):
    supports_images = False

    @abstractmethod
    def load(self, plan=None, progress=None) -> None: ...
    @abstractmethod
    def stream(self, messages: List[ChatMessage], cancel: threading.Event) -> Iterator[str]: ...
    def warmup(self, system_prompt: str) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------- in-process llama.cpp
class LlamaCppBackend(LLMBackend):
    def __init__(self, cfg: Config):
        self.cfg, self.llm = cfg, None

    def load(self, plan=None, progress=None) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise RuntimeError("llama-cpp-python is required: `pip install llama-cpp-python` "
                               "(or set llm.backend='llama_server' / 'openai').") from e
        path, _ = models.ensure_llm(self.cfg, progress)
        l = self.cfg.llm
        kw: Dict[str, Any] = dict(
            model_path=path, n_ctx=l.n_ctx, n_batch=l.n_batch,
            n_threads=(plan.llm_threads if plan else l.n_threads) or None,
            n_threads_batch=(plan.llm_batch_threads if plan else None) or None,
            n_gpu_layers=(plan.llm_gpu_layers if plan else 0),
            use_mmap=l.use_mmap, use_mlock=l.use_mlock, flash_attn=l.flash_attn, verbose=False)
        while True:                       # tolerate older llama-cpp-python builds that lack a kwarg
            try:
                self.llm = Llama(**kw)
                break
            except TypeError as e:
                bad = next((k for k in list(kw) if k in str(e)), None)
                if not bad or bad in ("model_path",):
                    raise
                kw.pop(bad)

    def warmup(self, system_prompt: str) -> None:
        """Pre-fill the KV cache with the system prompt so the first turn only pays for the user text."""
        if self.llm is None:
            return
        prefix = render_chatml([ChatMessage("system", system_prompt)], add_generation_prompt=False)
        toks = self.llm.tokenize(prefix.encode("utf-8"), add_bos=False, special=True)
        self.llm.reset()
        self.llm.eval(toks)

    def stream(self, messages: List[ChatMessage], cancel: threading.Event) -> Iterator[str]:
        assert self.llm is not None, "LLM not loaded"
        l = self.cfg.llm
        if any(m.images for m in messages):
            log.warning("in-process llama.cpp backend has no image support; use backend='llama_server'")
        prompt = render_chatml(messages, enable_thinking=l.enable_thinking, keep_empty_think=l.keep_empty_think_in_history)
        it = self.llm.create_completion(
            prompt, max_tokens=l.max_tokens, temperature=l.temperature, top_p=l.top_p, top_k=l.top_k,
            min_p=l.min_p, repeat_penalty=l.repeat_penalty, presence_penalty=l.presence_penalty,
            stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"], stream=True)
        try:
            for chunk in it:
                if cancel.is_set():
                    break
                piece = chunk["choices"][0].get("text", "")
                if piece:
                    yield piece
        finally:
            close = getattr(it, "close", None)
            if close:
                close()

    def close(self) -> None:
        self.llm = None


# ---------------------------------------------------------------- OpenAI-compatible
def _openai_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        if m.images:
            parts: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in m.images]
            parts.append({"type": "text", "text": m.content})
            out.append({"role": m.role, "content": parts})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class OpenAICompatBackend(LLMBackend):
    supports_images = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = cfg.llm.base_url.rstrip("/")

    def load(self, plan=None, progress=None) -> None: ...

    def _body(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        l = self.cfg.llm
        body: Dict[str, Any] = {
            "model": l.model_name, "messages": _openai_messages(messages), "stream": True,
            "max_tokens": l.max_tokens, "temperature": l.temperature, "top_p": l.top_p, "top_k": l.top_k,
            "min_p": l.min_p, "repeat_penalty": l.repeat_penalty, "presence_penalty": l.presence_penalty,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": l.enable_thinking}}
        body.update(l.extra_body)
        return body

    def stream(self, messages: List[ChatMessage], cancel: threading.Event) -> Iterator[str]:
        l = self.cfg.llm
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(self._body(messages)).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     **({"Authorization": f"Bearer {l.api_key}"} if l.api_key else {})})
        resp = urllib.request.urlopen(req, timeout=l.request_timeout_s)
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
                piece = delta.get("content")
                if piece:
                    yield piece
        finally:
            resp.close()


# ---------------------------------------------------------------- managed llama-server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServerBackend(OpenAICompatBackend):
    """Spawns ``llama-server`` (llama.cpp) - the most complete path: newest model
    architectures, image input via mmproj, GPU builds for every vendor."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.proc: Optional[subprocess.Popen] = None

    def load(self, plan=None, progress=None) -> None:
        l = self.cfg.llm
        binary = l.server_binary or shutil.which("llama-server") or shutil.which("llama-server.exe")
        if not binary:
            raise RuntimeError("llama-server not found. Install llama.cpp (winget install llama.cpp / brew install "
                               "llama.cpp) or set llm.server_binary.")
        want_vision = self.cfg.vision.enabled
        path, mmproj = models.ensure_llm(self.cfg, progress, need_mmproj=want_vision)
        port = l.server_port or _free_port()
        cmd = [binary, "-m", path, "-c", str(l.n_ctx), "-b", str(l.n_batch), "--host", "127.0.0.1",
               "--port", str(port), "--jinja"]
        threads = (plan.llm_threads if plan else l.n_threads)
        if threads:
            cmd += ["-t", str(threads)]
        if plan and plan.llm_gpu_layers:
            cmd += ["-ngl", "99" if plan.llm_gpu_layers < 0 else str(plan.llm_gpu_layers)]
        if want_vision and mmproj:
            cmd += ["--mmproj", mmproj]
        cmd += list(l.server_extra_args)
        log.info("starting llama-server: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base_url = f"http://127.0.0.1:{port}/v1"
        deadline = time.time() + max(60.0, l.request_timeout_s)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode}); "
                                   f"run it manually to see the error: {' '.join(cmd)}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.4)
        raise RuntimeError("llama-server did not become ready in time")

    def warmup(self, system_prompt: str) -> None:
        try:                                     # tiny request: primes the slot's prompt cache
            ev = threading.Event()
            msgs = [ChatMessage("system", system_prompt), ChatMessage("user", ".")]
            old = self.cfg.llm.max_tokens
            self.cfg.llm.max_tokens = 1
            try:
                for _ in self.stream(msgs, ev):
                    pass
            finally:
                self.cfg.llm.max_tokens = old
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


# ---------------------------------------------------------------- mock
class MockLLM(LLMBackend):
    """Scripted LLM for tests/demos.  ``replies`` may contain callables(messages)->str."""
    supports_images = True

    def __init__(self, replies=None, token_delay: float = 0.004, first_delay: float = 0.05):
        self.replies, self.i = list(replies or ["سلام! چطور می‌تونم کمکتون کنم؟"]), 0
        self.token_delay, self.first_delay = token_delay, first_delay
        self.seen: List[List[ChatMessage]] = []

    def load(self, plan=None, progress=None) -> None: ...

    def stream(self, messages: List[ChatMessage], cancel: threading.Event) -> Iterator[str]:
        self.seen.append(list(messages))
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        text = r(messages) if callable(r) else r
        time.sleep(self.first_delay)
        step = 3
        for k in range(0, len(text), step):
            if cancel.is_set():
                return
            time.sleep(self.token_delay)
            yield text[k:k + step]


def create_llm(cfg: Config) -> LLMBackend:
    b = cfg.llm.backend
    if b == "mock":
        return MockLLM()
    if cfg.vision.enabled and b == "llama_cpp":
        if shutil.which("llama-server") or cfg.llm.server_binary:
            log.info("vision enabled -> using llama-server backend")
            return LlamaServerBackend(cfg)
        log.warning("vision requested but llama-server not found; images will be ignored")
    if b == "llama_cpp":
        return LlamaCppBackend(cfg)
    if b == "llama_server":
        return LlamaServerBackend(cfg)
    if b == "openai":
        return OpenAICompatBackend(cfg)
    raise ValueError(f"unknown llm.backend {b!r}")
