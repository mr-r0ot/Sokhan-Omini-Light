"""Function calling for any backend.

Tools are described to the model in its system prompt and called through the
``<tool_call>`` text protocol, so they work with every LLM backend, local or
remote, without server-side tool support.

    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        '''Current weather for a city.

        city: city name, e.g. "Tehran"
        '''
        return "sunny, 24 degrees"

    omni = Omni(tools=[get_weather])
"""
from __future__ import annotations

import inspect
import json
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

_PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema(tp: Any) -> Dict[str, Any]:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Union:
        rest = [a for a in args if a is not type(None)]
        return _schema(rest[0]) if len(rest) == 1 else {}
    if origin is typing.Literal:
        return {"type": _PY2JSON.get(type(args[0]), "string"), "enum": list(args)}
    if origin in (list, List):
        return {"type": "array", "items": _schema(args[0]) if args else {}}
    if origin in (dict, Dict):
        return {"type": "object"}
    return {"type": _PY2JSON.get(tp, "string")}


def _param_docs(doc: str) -> Dict[str, str]:
    """Lines like ``name: description`` (Google style ``Args:`` blocks included)."""
    out = {}
    for line in doc.splitlines():
        m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
        if m and m.group(1).lower() not in ("args", "returns", "raises", "example", "examples"):
            out[m.group(1)] = m.group(2).strip()
    return out


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    @classmethod
    def from_function(cls, fn: Callable[..., Any], description: str = "", name: Optional[str] = None) -> "Tool":
        doc = inspect.getdoc(fn) or ""
        summary = doc.split("\n\n")[0].strip()
        pdocs = _param_docs(doc)
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = {}
        props, required = {}, []
        for pname, p in inspect.signature(fn).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            s = _schema(hints.get(pname, str))
            if pname in pdocs:
                s["description"] = pdocs[pname]
            props[pname] = s
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return cls(name or fn.__name__, description or summary, fn,
                   {"type": "object", "properties": props, "required": required})

    def spec(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None, description: str = ""):
    """Decorator: ``@tool`` or ``@tool(name="...", description="...")``."""
    def wrap(f: Callable) -> Tool:
        return Tool.from_function(f, description, name)
    return wrap(fn) if fn is not None else wrap


class ToolRegistry:
    def __init__(self, tools: Optional[List[Union[Tool, Callable]]] = None):
        self.tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Union[Tool, Callable]) -> Tool:
        t = t if isinstance(t, Tool) else Tool.from_function(t)
        self.tools[t.name] = t
        return t

    def __bool__(self) -> bool:
        return bool(self.tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        t = self.tools.get(name)
        if t is None:
            return json.dumps({"error": f"unknown function {name}"})
        try:
            res = t.fn(**arguments)
        except Exception as e:                     # a buggy tool must never break the conversation
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
