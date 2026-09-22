"""Tiny tool/function-calling layer that works with *any* backend by using the
Qwen-style ``<tool_call>`` text protocol (no server-side tool support needed)."""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    @classmethod
    def from_function(cls, fn: Callable[..., Any], description: str = "", name: Optional[str] = None) -> "Tool":
        sig = inspect.signature(fn)
        props, req = {}, []
        for pname, p in sig.parameters.items():
            props[pname] = {"type": _PY2JSON.get(p.annotation, "string")}
            if p.default is inspect._empty:
                req.append(pname)
        return cls(name or fn.__name__, description or (fn.__doc__ or "").strip(),
                   fn, {"type": "object", "properties": props, "required": req})

    def spec(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def __bool__(self) -> bool:
        return bool(self.tools)

    def prompt_block(self) -> str:
        if not self.tools:
            return ""
        specs = "\n".join(json.dumps(t.spec(), ensure_ascii=False) for t in self.tools.values())
        return ("\n\n# Tools\nYou may call functions to help the user. Function signatures are inside <tools></tools>:\n"
                f"<tools>\n{specs}\n</tools>\n"
                "To call a function, output exactly:\n<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'
                "Call a function only when needed, then wait for the result before answering. "
                "Never speak the tool call itself.")

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
        try:
            res = tool.fn(**arguments)
        except Exception as e:  # tool bugs must never crash the conversation
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
