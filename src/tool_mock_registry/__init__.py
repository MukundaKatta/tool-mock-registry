"""
tool_mock_registry — Mock tool registry for testing agent tool calls.

Register mock implementations per tool name, call them through the registry,
and verify call counts. Unit-test agent logic without hitting real tool endpoints.
"""

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolNotFoundError(Exception):
    """Raised when a tool name is not registered in the registry."""


@dataclass
class CallRecord:
    """Record of a single tool invocation."""

    tool_name: str
    kwargs: dict
    result: object
    error: str | None = None


class MockRegistry:
    """Thread-safe registry of mock tool implementations.

    Usage::

        registry = MockRegistry()
        registry.register_return("search", {"results": []})
        result = registry.call("search", query="cats")
        registry.assert_called("search", times=1)

    Can also be used as a context manager; ``__exit__`` resets counts/history
    while keeping the registered mocks in place.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # tool_name -> callable
        self._mocks: dict[str, Callable] = {}
        # tool_name -> call count
        self._counts: dict[str, int] = {}
        # flat list of all call records
        self._history: list[CallRecord] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_mock(self, tool_name: str, fn: Callable) -> "MockRegistry":
        """Register a callable as the implementation for *tool_name*.

        Returns self so calls can be chained.
        """
        with self._lock:
            self._mocks[tool_name] = fn
            # Ensure counter entry exists (don't reset if already present)
            self._counts.setdefault(tool_name, 0)
        return self

    def register_return(self, tool_name: str, value: Any) -> "MockRegistry":
        """Register a mock that always returns *value* for *tool_name*."""
        return self.register_mock(tool_name, lambda **_kw: value)

    def register_error(self, tool_name: str, exc: Exception) -> "MockRegistry":
        """Register a mock that always raises *exc* for *tool_name*."""

        def _raise(**_kw: Any) -> None:
            raise exc

        return self.register_mock(tool_name, _raise)

    def is_mocked(self, tool_name: str) -> bool:
        """Return True if *tool_name* has a registered mock."""
        with self._lock:
            return tool_name in self._mocks

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke the mock registered for *tool_name* with the given kwargs.

        Raises:
            ToolNotFoundError: if *tool_name* has no registered mock.
            Exception: whatever the mock raises (after recording the error).
        """
        with self._lock:
            if tool_name not in self._mocks:
                raise ToolNotFoundError(f"No mock registered for tool '{tool_name}'")
            fn = self._mocks[tool_name]

        # Call outside the lock so recursive calls don't deadlock
        try:
            result = fn(**kwargs)
            record = CallRecord(tool_name=tool_name, kwargs=kwargs, result=result)
            with self._lock:
                self._counts[tool_name] = self._counts.get(tool_name, 0) + 1
                self._history.append(record)
            return result
        except Exception as exc:
            record = CallRecord(
                tool_name=tool_name,
                kwargs=kwargs,
                result=None,
                error=str(exc),
            )
            with self._lock:
                self._counts[tool_name] = self._counts.get(tool_name, 0) + 1
                self._history.append(record)
            raise

    async def call_async(self, tool_name: str, **kwargs: Any) -> Any:
        """Async-aware invocation: awaits the mock if it is a coroutine function.

        Raises:
            ToolNotFoundError: if *tool_name* has no registered mock.
        """
        with self._lock:
            if tool_name not in self._mocks:
                raise ToolNotFoundError(f"No mock registered for tool '{tool_name}'")
            fn = self._mocks[tool_name]

        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = fn(**kwargs)
            record = CallRecord(tool_name=tool_name, kwargs=kwargs, result=result)
            with self._lock:
                self._counts[tool_name] = self._counts.get(tool_name, 0) + 1
                self._history.append(record)
            return result
        except Exception as exc:
            record = CallRecord(
                tool_name=tool_name,
                kwargs=kwargs,
                result=None,
                error=str(exc),
            )
            with self._lock:
                self._counts[tool_name] = self._counts.get(tool_name, 0) + 1
                self._history.append(record)
            raise

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def call_count(self, tool_name: str) -> int:
        """Return how many times *tool_name* has been called (0 if never)."""
        with self._lock:
            return self._counts.get(tool_name, 0)

    def call_counts(self) -> dict[str, int]:
        """Return a copy of the call-count dict for all registered mocks."""
        with self._lock:
            return dict(self._counts)

    def call_history(self, tool_name: str | None = None) -> list[CallRecord]:
        """Return a copy of call records, optionally filtered by *tool_name*.

        Pass ``None`` (default) to get all records.
        """
        with self._lock:
            if tool_name is None:
                return list(self._history)
            return [r for r in self._history if r.tool_name == tool_name]

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_called(self, tool_name: str, times: int | None = None) -> None:
        """Assert that *tool_name* was called at least once (or exactly *times*).

        Raises AssertionError on failure.
        """
        count = self.call_count(tool_name)
        if times is None:
            if count == 0:
                raise AssertionError(f"Expected '{tool_name}' to be called but it was not.")
        else:
            if count != times:
                raise AssertionError(
                    f"Expected '{tool_name}' to be called {times} time(s)"
                    f" but was called {count} time(s)."
                )

    def assert_not_called(self, tool_name: str) -> None:
        """Assert that *tool_name* was never called.

        Raises AssertionError if it was called at least once.
        """
        count = self.call_count(tool_name)
        if count > 0:
            raise AssertionError(
                f"Expected '{tool_name}' NOT to be called but it was called {count} time(s)."
            )

    # ------------------------------------------------------------------
    # Reset / clear
    # ------------------------------------------------------------------

    def reset_counts(self) -> None:
        """Clear call history and counts, but keep registered mocks."""
        with self._lock:
            self._history.clear()
            self._counts = {name: 0 for name in self._mocks}

    def clear_mocks(self, tool_name: str | None = None) -> None:
        """Remove registered mock(s).

        Pass a *tool_name* to remove only that mock; pass ``None`` to remove all.
        """
        with self._lock:
            if tool_name is None:
                self._mocks.clear()
                self._counts.clear()
                self._history.clear()
            else:
                self._mocks.pop(tool_name, None)
                self._counts.pop(tool_name, None)
                self._history = [r for r in self._history if r.tool_name != tool_name]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MockRegistry":
        return self

    def __exit__(self, *args: Any) -> None:
        """Reset counts and history on exit (mocks stay registered)."""
        self.reset_counts()


__all__ = ["MockRegistry", "CallRecord", "ToolNotFoundError"]
