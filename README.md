# tool-mock-registry

Mock tool registry for testing agent tool calls. Register mock implementations per tool name, call them through the registry, and verify call counts. Unit-test agent logic without hitting real tool endpoints.

Zero runtime dependencies. Python 3.10+.

## Install

```bash
pip install tool-mock-registry
```

## Quick start

```python
from tool_mock_registry import MockRegistry

registry = MockRegistry()
registry.register_return("search", {"results": ["cat", "dog"]})
registry.register_mock("summarize", lambda **kw: f"Summary of: {kw['text']}")

result = registry.call("search", query="animals")
registry.assert_called("search", times=1)
registry.assert_not_called("summarize")
```

## API

### Registration

| Method | Description |
|---|---|
| `register_mock(name, fn)` | Register a callable; returns `self` for chaining |
| `register_return(name, value)` | Always return `value` |
| `register_error(name, exc)` | Always raise `exc` |
| `is_mocked(name)` | Check if a tool is registered |

### Invocation

| Method | Description |
|---|---|
| `call(name, **kwargs)` | Invoke mock synchronously; raises `ToolNotFoundError` if not registered |
| `await call_async(name, **kwargs)` | Awaits the mock if it is a coroutine function; otherwise calls it synchronously |

### Inspection

| Method | Description |
|---|---|
| `call_count(name)` | Number of calls for one tool (0 if never called) |
| `call_counts()` | Dict of all call counts |
| `call_history(name=None)` | List of `CallRecord`; pass `None` for all tools |

### Assertions

| Method | Description |
|---|---|
| `assert_called(name, times=None)` | Fails if never called; with `times`, checks exact count |
| `assert_not_called(name)` | Fails if called at least once |

### Reset / clear

| Method | Description |
|---|---|
| `reset_counts()` | Clear history + counts; keep mocks |
| `clear_mocks(name=None)` | Remove one mock or all |

### Context manager

`__exit__` calls `reset_counts()` (mocks stay registered).

```python
with MockRegistry() as reg:
    reg.register_return("tool", 42)
    reg.call("tool")
    reg.assert_called("tool")
# counts/history reset; mock still present
```

## Thread safety

All methods are protected by a re-entrant lock and safe for concurrent use.

## License

MIT
