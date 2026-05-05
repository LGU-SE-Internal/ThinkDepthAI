"""Adapter to convert AsyncBaseToolkit methods into LangChain tools."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Any

from .base import AsyncBaseToolkit

_WRAPPER_KEYS = {"args", "kwargs", "arguments", "parameters", "params", "input"}
_META_KEYS = {"config", "run_name", "run_id", "metadata", "tags", "callbacks"}


def toolkit_to_langchain_tools(toolkit: AsyncBaseToolkit) -> list[Any]:
    """Convert all @register_tool methods on toolkit to LangChain StructuredTools."""
    from langchain_core.tools import StructuredTool

    tools_map = toolkit.get_tools_map_func()
    lc_tools: list[Any] = []
    for tool_name, method in tools_map.items():
        lc_tool = _method_to_langchain_tool(tool_name, method, StructuredTool)
        lc_tools.append(lc_tool)
    return lc_tools


def _method_to_langchain_tool(tool_name: str, method: Callable[..., Any], structured_tool_cls: Any) -> Any:
    sig = inspect.signature(method)
    valid_params = {name for name, p in sig.parameters.items() if name != "self"}
    description = inspect.getdoc(method) or f"Tool: {tool_name}"

    async def _invoke(**kwargs: Any) -> str:
        try:
            kwargs = _unwrap_llm_args(kwargs, valid_params, tool_name)
            result = await method(**kwargs)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except TypeError as e:
            return json.dumps(
                {
                    "error": f"Invalid parameters: {e}",
                    "tool": tool_name,
                    "accepted_parameters": sorted(valid_params),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}", "tool": tool_name}, ensure_ascii=False)

    param_docs = _parse_docstring_args(inspect.getdoc(method))
    schema = _signature_to_json_schema(sig, tool_name, param_docs)

    return structured_tool_cls.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=description,
        args_schema=schema,
        func=None,
        infer_schema=False,
    )


def _unwrap_llm_args(kwargs: dict[str, Any], valid_params: set[str], tool_name: str) -> dict[str, Any]:
    if valid_params and set(kwargs.keys()) <= valid_params:
        return kwargs
    for meta_key in _META_KEYS:
        kwargs.pop(meta_key, None)
    if valid_params and set(kwargs.keys()) <= valid_params:
        return kwargs
    for wrapper_key in _WRAPPER_KEYS:
        if wrapper_key in kwargs and isinstance(kwargs[wrapper_key], dict):
            inner = kwargs[wrapper_key]
            if valid_params and (set(inner.keys()) & valid_params):
                unwrapped = {k: v for k, v in kwargs.items() if k not in _WRAPPER_KEYS and k not in _META_KEYS}
                unwrapped.update(inner)
                return {k: v for k, v in unwrapped.items() if k in valid_params or not valid_params}
    if valid_params:
        unknown_keys = set(kwargs.keys()) - valid_params
        if unknown_keys:
            cleaned = {k: v for k, v in kwargs.items() if k in valid_params}
            if cleaned:
                return cleaned
    return kwargs


_PYTHON_TYPE_TO_JSON: dict[type | str, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _signature_to_json_schema(
    sig: inspect.Signature, tool_name: str, param_docs: dict[str, str] | None = None
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    param_docs = param_docs or {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        prop: dict[str, Any] = {}
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            prop["type"] = "string"
        else:
            prop.update(_annotation_to_schema(annotation))
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return {"type": "object", "properties": properties, "required": required}


def _parse_docstring_args(docstring: str | None) -> dict[str, str]:
    if not docstring:
        return {}
    args_match = re.search(r"\bArgs:\s*\n(.*?)(?:\n\s*\n|\n\s*Returns:|\n\s*Raises:|\Z)", docstring, re.DOTALL)
    if not args_match:
        return {}
    args_text = args_match.group(1)
    result: dict[str, str] = {}
    current_param = None
    current_desc_parts: list[str] = []
    for line in args_text.split("\n"):
        param_match = re.match(r"\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if param_match:
            if current_param:
                result[current_param] = " ".join(current_desc_parts).strip()
            current_param = param_match.group(1)
            current_desc_parts = [param_match.group(2).strip()] if param_match.group(2).strip() else []
        elif current_param and line.strip():
            current_desc_parts.append(line.strip())
    if current_param:
        result[current_param] = " ".join(current_desc_parts).strip()
    return result


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    import typing

    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = annotation.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}
    if origin is list:
        args = getattr(annotation, "__args__", None)
        if args:
            return {"type": "array", "items": _annotation_to_schema(args[0])}
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    json_type = _PYTHON_TYPE_TO_JSON.get(annotation)
    if json_type:
        return {"type": json_type}
    return {"type": "string"}
