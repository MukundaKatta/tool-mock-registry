"""
tool-mock-registry: Mock tool registry for testing agent tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class ToolNotRegistered(KeyError):
    pass


class AssertionFailed(AssertionError):
    pass


@dataclass
class CallRecord:
    tool_name: str
    args: dict[str, Any]
    result: Any
    raised: Optional[BaseException] = None


class MockTool:
    """A single registered mock tool."""

    def __init__(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Optional[Callable[..., Any]] = None,
        raise_on_call: Optional[BaseException] = None,
    ) -> None:
        self.name = name
        self._return_value = return_value
        self._side_effect = side_effect
        self._raise_on_call = raise_on_call
        self._calls: list[CallRecord] = []

    def __call__(self, **kwargs: Any) -> Any:
        if self._raise_on_call is not None:
            self._calls.append(CallRecord(tool_name=self.name, args=kwargs, result=None, raised=self._raise_on_call))
            raise self._raise_on_call
        if self._side_effect is not None:
            result = self._side_effect(**kwargs)
        else:
            result = self._return_value
        self._calls.append(CallRecord(tool_name=self.name, args=kwargs, result=result))
        return result

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def calls(self) -> list[CallRecord]:
        return list(self._calls)

    @property
    def last_call(self) -> Optional[CallRecord]:
        return self._calls[-1] if self._calls else None

    def assert_called(self) -> None:
        if not self._calls:
            raise AssertionFailed(f"Expected '{self.name}' to have been called, but it was not.")

    def assert_called_once(self) -> None:
        if self.call_count != 1:
            raise AssertionFailed(f"Expected '{self.name}' to be called once, but called {self.call_count} times.")

    def assert_called_with(self, **expected_kwargs: Any) -> None:
        if not self._calls:
            raise AssertionFailed(f"'{self.name}' was never called.")
        last = self._calls[-1].args
        for k, v in expected_kwargs.items():
            if last.get(k) != v:
                raise AssertionFailed(f"'{self.name}' last called with {k}={last.get(k)!r}, expected {v!r}.")

    def assert_not_called(self) -> None:
        if self._calls:
            raise AssertionFailed(f"Expected '{self.name}' not to be called, but called {self.call_count} times.")

    def reset(self) -> None:
        self._calls.clear()


class MockToolRegistry:
    """
    Registry of mock tools for testing agent tool-calling flows.

    Usage::

        registry = MockToolRegistry()
        registry.register("search_web", return_value=["result1"])
        registry.register("send_email", side_effect=lambda to, body: True)

        result = registry.call("search_web", query="python")
        assert result == ["result1"]

        registry.get("search_web").assert_called_once()
    """

    def __init__(self) -> None:
        self._tools: dict[str, MockTool] = {}

    def register(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Optional[Callable[..., Any]] = None,
        raise_on_call: Optional[BaseException] = None,
    ) -> MockTool:
        tool = MockTool(name=name, return_value=return_value, side_effect=side_effect, raise_on_call=raise_on_call)
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> MockTool:
        if name not in self._tools:
            raise ToolNotRegistered(name)
        return self._tools[name]

    def call(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def has(self, name: str) -> bool:
        return name in self._tools

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def reset_all(self) -> None:
        for tool in self._tools.values():
            tool.reset()

    def all_calls(self) -> list[CallRecord]:
        result: list[CallRecord] = []
        for tool in self._tools.values():
            result.extend(tool.calls)
        return result

    def assert_tool_called(self, name: str) -> None:
        self.get(name).assert_called()

    def assert_tool_not_called(self, name: str) -> None:
        self.get(name).assert_not_called()

    def __enter__(self) -> "MockToolRegistry":
        return self

    def __exit__(self, *args: Any) -> None:
        self.reset_all()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


__all__ = ["MockToolRegistry", "MockTool", "CallRecord", "ToolNotRegistered", "AssertionFailed"]
