"""Toolkit base classes and utilities."""

from collections.abc import Callable
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


@overload
def register_tool[F: Callable[..., Any]](name_or_func: F) -> F: ...


@overload
def register_tool(name_or_func: str | None = None) -> Callable[[F], F]: ...


def register_tool[F: Callable[..., Any]](name_or_func: str | F | None = None) -> F | Callable[[F], F]:
    """Decorator to register a method as a tool.

    Usage:
        @register_tool  # uses method name
        @register_tool()  # uses method name
        @register_tool("custom_name")  # uses custom name
    """

    def decorator(func: F) -> F:
        if isinstance(name_or_func, str):
            tool_name = name_or_func
        else:
            tool_name = func.__name__
        func._is_tool = True  # type: ignore[attr-defined]
        func._tool_name = tool_name  # type: ignore[attr-defined]
        return func

    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator


class AsyncBaseToolkit:
    """Base class for toolkits."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._tools_map: dict[str, Callable[..., Any]] | None = None

    @property
    def tools_map(self) -> dict[str, Callable]:
        if self._tools_map is None:
            self._tools_map = {}
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if callable(attr) and getattr(attr, "_is_tool", False):
                    tool_name: str = attr._tool_name
                    self._tools_map[tool_name] = attr
        return self._tools_map

    def get_tools_map_func(self) -> dict[str, Callable]:
        """Get tools map, optionally filtered by activated_tools config."""
        activated = self.config.get("activated_tools") if isinstance(self.config, dict) else None
        if activated:
            return {name: self.tools_map[name] for name in activated if name in self.tools_map}
        return self.tools_map

    async def call_tool(self, name: str, arguments: dict) -> str:
        tools_map = self.get_tools_map_func()
        if name not in tools_map:
            raise ValueError(f"Tool {name} not found")
        return await tools_map[name](**arguments)
