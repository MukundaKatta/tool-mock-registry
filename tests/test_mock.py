"""Tests for tool_mock_registry.MockRegistry (~30 cases)."""

import threading

import pytest

from tool_mock_registry import (
    CallRecord,
    MockRegistry,
    ToolNotFoundError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_registry(*tool_names: str, return_value: object = "ok") -> MockRegistry:
    r = MockRegistry()
    for name in tool_names:
        r.register_return(name, return_value)
    return r


# ---------------------------------------------------------------------------
# 1. register_mock() + call(): fn called with correct kwargs
# ---------------------------------------------------------------------------


def test_register_mock_calls_fn_with_kwargs():
    received: list[dict] = []
    r = MockRegistry()
    r.register_mock("search", lambda **kw: received.append(kw) or "done")
    result = r.call("search", query="cats", limit=5)
    assert result == "done"
    assert received == [{"query": "cats", "limit": 5}]


# ---------------------------------------------------------------------------
# 2. register_return(): always returns value
# ---------------------------------------------------------------------------


def test_register_return_always_returns_value():
    r = MockRegistry()
    r.register_return("fetch", {"data": [1, 2, 3]})
    for _ in range(3):
        assert r.call("fetch") == {"data": [1, 2, 3]}


def test_register_return_ignores_kwargs():
    r = MockRegistry()
    r.register_return("noop", 42)
    assert r.call("noop", x=1, y=2) == 42


# ---------------------------------------------------------------------------
# 3. register_error(): always raises specified exception
# ---------------------------------------------------------------------------


def test_register_error_always_raises():
    r = MockRegistry()
    exc = ValueError("boom")
    r.register_error("bad_tool", exc)
    with pytest.raises(ValueError, match="boom"):
        r.call("bad_tool")


def test_register_error_raises_each_call():
    r = MockRegistry()
    r.register_error("flaky", RuntimeError("oops"))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            r.call("flaky")


# ---------------------------------------------------------------------------
# 4. is_mocked()
# ---------------------------------------------------------------------------


def test_is_mocked_true_after_register():
    r = MockRegistry()
    r.register_return("tool_a", None)
    assert r.is_mocked("tool_a") is True


def test_is_mocked_false_for_unknown():
    r = MockRegistry()
    assert r.is_mocked("ghost") is False


def test_is_mocked_false_after_clear():
    r = MockRegistry()
    r.register_return("tool_b", 0)
    r.clear_mocks("tool_b")
    assert r.is_mocked("tool_b") is False


# ---------------------------------------------------------------------------
# 5. call(): ToolNotFoundError for unknown tool
# ---------------------------------------------------------------------------


def test_call_unknown_tool_raises_tool_not_found():
    r = MockRegistry()
    with pytest.raises(ToolNotFoundError):
        r.call("nonexistent")


def test_call_unknown_error_message_contains_name():
    r = MockRegistry()
    with pytest.raises(ToolNotFoundError, match="nonexistent"):
        r.call("nonexistent")


# ---------------------------------------------------------------------------
# 6. call(): records success in history
# ---------------------------------------------------------------------------


def test_call_records_success_in_history():
    r = MockRegistry()
    r.register_return("calc", 99)
    r.call("calc", x=1)
    history = r.call_history("calc")
    assert len(history) == 1
    assert history[0].tool_name == "calc"
    assert history[0].kwargs == {"x": 1}
    assert history[0].result == 99
    assert history[0].error is None


# ---------------------------------------------------------------------------
# 7. call(): records error in history, still re-raises
# ---------------------------------------------------------------------------


def test_call_records_error_in_history_and_reraises():
    r = MockRegistry()
    r.register_error("kaboom", TypeError("type mismatch"))
    with pytest.raises(TypeError):
        r.call("kaboom", val="bad")

    history = r.call_history("kaboom")
    assert len(history) == 1
    assert history[0].error == "type mismatch"
    assert history[0].result is None


# ---------------------------------------------------------------------------
# 8. call_async(): sync mock works
# ---------------------------------------------------------------------------


async def test_call_async_sync_mock():
    r = MockRegistry()
    r.register_return("ping", "pong")
    result = await r.call_async("ping", target="server")
    assert result == "pong"
    assert r.call_count("ping") == 1


# ---------------------------------------------------------------------------
# 9. call_async(): async mock awaited
# ---------------------------------------------------------------------------


async def test_call_async_awaits_coroutine_mock():
    async def async_lookup(**kw):
        return f"found:{kw.get('key')}"

    r = MockRegistry()
    r.register_mock("lookup", async_lookup)
    result = await r.call_async("lookup", key="abc")
    assert result == "found:abc"
    assert r.call_count("lookup") == 1


async def test_call_async_records_history_for_coroutine():
    async def async_fn(**_kw):
        return "async_result"

    r = MockRegistry()
    r.register_mock("async_tool", async_fn)
    await r.call_async("async_tool", n=7)
    history = r.call_history("async_tool")
    assert len(history) == 1
    assert history[0].result == "async_result"


# ---------------------------------------------------------------------------
# 10. call_count(): 0 before calls, increments
# ---------------------------------------------------------------------------


def test_call_count_zero_before_any_call():
    r = MockRegistry()
    r.register_return("tool", None)
    assert r.call_count("tool") == 0


def test_call_count_increments():
    r = MockRegistry()
    r.register_return("counter_tool", "x")
    r.call("counter_tool")
    r.call("counter_tool")
    r.call("counter_tool")
    assert r.call_count("counter_tool") == 3


def test_call_count_unknown_tool_returns_zero():
    r = MockRegistry()
    assert r.call_count("never_registered") == 0


# ---------------------------------------------------------------------------
# 11. call_counts(): dict of all
# ---------------------------------------------------------------------------


def test_call_counts_returns_all_registered_tools():
    r = MockRegistry()
    r.register_return("a", 1)
    r.register_return("b", 2)
    r.call("a")
    r.call("a")
    r.call("b")
    counts = r.call_counts()
    assert counts["a"] == 2
    assert counts["b"] == 1


def test_call_counts_is_a_copy():
    r = MockRegistry()
    r.register_return("t", None)
    counts = r.call_counts()
    counts["t"] = 999  # mutate the copy
    assert r.call_count("t") == 0  # original unaffected


# ---------------------------------------------------------------------------
# 12. call_history(name): filtered; call_history(None): all
# ---------------------------------------------------------------------------


def test_call_history_filtered_by_name():
    r = MockRegistry()
    r.register_return("x", "rx")
    r.register_return("y", "ry")
    r.call("x")
    r.call("y")
    r.call("x")
    assert len(r.call_history("x")) == 2
    assert len(r.call_history("y")) == 1
    assert all(rec.tool_name == "x" for rec in r.call_history("x"))


def test_call_history_none_returns_all():
    r = MockRegistry()
    r.register_return("p", 1)
    r.register_return("q", 2)
    r.call("p")
    r.call("q")
    assert len(r.call_history()) == 2
    assert len(r.call_history(None)) == 2


# ---------------------------------------------------------------------------
# 13. call_history() returns copy
# ---------------------------------------------------------------------------


def test_call_history_returns_copy():
    r = MockRegistry()
    r.register_return("tool", None)
    r.call("tool")
    copy1 = r.call_history()
    copy1.append(CallRecord(tool_name="fake", kwargs={}, result=None))
    assert len(r.call_history()) == 1  # original unaffected


# ---------------------------------------------------------------------------
# 14. assert_called(): passes when called, AssertionError when not
# ---------------------------------------------------------------------------


def test_assert_called_passes_when_called():
    r = MockRegistry()
    r.register_return("svc", "ok")
    r.call("svc")
    r.assert_called("svc")  # should not raise


def test_assert_called_fails_when_not_called():
    r = MockRegistry()
    r.register_return("svc", "ok")
    with pytest.raises(AssertionError):
        r.assert_called("svc")


# ---------------------------------------------------------------------------
# 15. assert_called(times=N): exact count check
# ---------------------------------------------------------------------------


def test_assert_called_times_exact_match():
    r = MockRegistry()
    r.register_return("fn", None)
    r.call("fn")
    r.call("fn")
    r.assert_called("fn", times=2)  # exact match — should not raise


def test_assert_called_times_wrong_count_raises():
    r = MockRegistry()
    r.register_return("fn", None)
    r.call("fn")
    with pytest.raises(AssertionError, match="1 time"):
        r.assert_called("fn", times=3)


# ---------------------------------------------------------------------------
# 16. assert_not_called(): passes when not called, fails when called
# ---------------------------------------------------------------------------


def test_assert_not_called_passes_for_uncalled_tool():
    r = MockRegistry()
    r.register_return("quiet", None)
    r.assert_not_called("quiet")  # should not raise


def test_assert_not_called_fails_when_called():
    r = MockRegistry()
    r.register_return("loud", None)
    r.call("loud")
    with pytest.raises(AssertionError):
        r.assert_not_called("loud")


# ---------------------------------------------------------------------------
# 17. reset_counts(): clears history, keeps mocks registered
# ---------------------------------------------------------------------------


def test_reset_counts_clears_history_and_counts():
    r = MockRegistry()
    r.register_return("op", "val")
    r.call("op")
    r.call("op")
    r.reset_counts()
    assert r.call_count("op") == 0
    assert r.call_history("op") == []


def test_reset_counts_keeps_mocks_registered():
    r = MockRegistry()
    r.register_return("op", "val")
    r.call("op")
    r.reset_counts()
    # mock is still there
    assert r.is_mocked("op")
    result = r.call("op")
    assert result == "val"
    assert r.call_count("op") == 1


# ---------------------------------------------------------------------------
# 18. clear_mocks(name): removes that tool
# ---------------------------------------------------------------------------


def test_clear_mocks_by_name_removes_only_that_tool():
    r = MockRegistry()
    r.register_return("a", 1)
    r.register_return("b", 2)
    r.clear_mocks("a")
    assert not r.is_mocked("a")
    assert r.is_mocked("b")


def test_clear_mocks_by_name_raises_on_subsequent_call():
    r = MockRegistry()
    r.register_return("gone", 0)
    r.clear_mocks("gone")
    with pytest.raises(ToolNotFoundError):
        r.call("gone")


# ---------------------------------------------------------------------------
# 19. clear_mocks(): removes all
# ---------------------------------------------------------------------------


def test_clear_mocks_all_removes_everything():
    r = MockRegistry()
    r.register_return("x", 1)
    r.register_return("y", 2)
    r.clear_mocks()
    assert not r.is_mocked("x")
    assert not r.is_mocked("y")
    assert r.call_counts() == {}
    assert r.call_history() == []


# ---------------------------------------------------------------------------
# 20. Context manager __exit__ calls reset_counts
# ---------------------------------------------------------------------------


def test_context_manager_resets_on_exit():
    r = MockRegistry()
    r.register_return("ctx_tool", "yes")
    with r:
        r.call("ctx_tool")
        assert r.call_count("ctx_tool") == 1
    # after exit: counts reset, mock still present
    assert r.call_count("ctx_tool") == 0
    assert r.is_mocked("ctx_tool")


def test_context_manager_returns_registry():
    r = MockRegistry()
    with r as reg:
        assert reg is r


# ---------------------------------------------------------------------------
# 21. register_mock() returns self (chaining)
# ---------------------------------------------------------------------------


def test_register_mock_returns_self_for_chaining():
    r = MockRegistry()
    result = r.register_mock("t1", lambda **_: 1).register_return("t2", 2).register_error(
        "t3", ValueError("e")
    )
    assert result is r
    assert r.is_mocked("t1")
    assert r.is_mocked("t2")
    assert r.is_mocked("t3")


# ---------------------------------------------------------------------------
# 22. Thread safety: 10 threads each call same mock, correct total count
# ---------------------------------------------------------------------------


def test_thread_safety_call_count():
    r = MockRegistry()
    r.register_return("shared", "ok")

    errors: list[Exception] = []

    def worker():
        try:
            for _ in range(100):
                r.call("shared")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert r.call_count("shared") == 1000


def test_thread_safety_history_length():
    r = MockRegistry()
    r.register_return("item", None)

    def worker():
        for _ in range(50):
            r.call("item")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(r.call_history()) == 500


# ---------------------------------------------------------------------------
# Bonus: async error recording
# ---------------------------------------------------------------------------


async def test_call_async_records_error_for_async_mock():
    async def broken(**_kw):
        raise KeyError("missing")

    r = MockRegistry()
    r.register_mock("async_err", broken)
    with pytest.raises(KeyError):
        await r.call_async("async_err")

    history = r.call_history("async_err")
    assert len(history) == 1
    assert history[0].error == "'missing'"
