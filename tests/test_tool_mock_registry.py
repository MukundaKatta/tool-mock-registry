"""Tests for tool-mock-registry."""
import pytest
from tool_mock_registry import MockToolRegistry, MockTool, CallRecord, ToolNotRegistered, AssertionFailed


def test_register_and_call():
    reg = MockToolRegistry()
    reg.register("search", return_value=["result"])
    result = reg.call("search", query="python")
    assert result == ["result"]


def test_call_unregistered():
    reg = MockToolRegistry()
    with pytest.raises(ToolNotRegistered):
        reg.call("missing_tool")


def test_side_effect():
    reg = MockToolRegistry()
    reg.register("add", side_effect=lambda x, y: x + y)
    assert reg.call("add", x=2, y=3) == 5


def test_raise_on_call():
    reg = MockToolRegistry()
    reg.register("bad_tool", raise_on_call=ValueError("oops"))
    with pytest.raises(ValueError, match="oops"):
        reg.call("bad_tool")


def test_call_count():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.call("tool")
    reg.call("tool")
    assert reg.get("tool").call_count == 2


def test_assert_called():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.call("tool")
    reg.assert_tool_called("tool")  # should not raise


def test_assert_called_fails_when_not_called():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    with pytest.raises(AssertionFailed):
        reg.assert_tool_called("tool")


def test_assert_not_called():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.assert_tool_not_called("tool")


def test_assert_not_called_fails():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.call("tool")
    with pytest.raises(AssertionFailed):
        reg.assert_tool_not_called("tool")


def test_assert_called_once():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.call("tool")
    reg.get("tool").assert_called_once()


def test_assert_called_once_fails():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.call("tool")
    reg.call("tool")
    with pytest.raises(AssertionFailed):
        reg.get("tool").assert_called_once()


def test_assert_called_with():
    reg = MockToolRegistry()
    reg.register("search", return_value=[])
    reg.call("search", query="python")
    reg.get("search").assert_called_with(query="python")


def test_assert_called_with_wrong_arg():
    reg = MockToolRegistry()
    reg.register("search", return_value=[])
    reg.call("search", query="python")
    with pytest.raises(AssertionFailed):
        reg.get("search").assert_called_with(query="java")


def test_reset_all():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    reg.call("tool")
    reg.reset_all()
    assert reg.get("tool").call_count == 0


def test_all_calls():
    reg = MockToolRegistry()
    reg.register("a", return_value=1)
    reg.register("b", return_value=2)
    reg.call("a")
    reg.call("b")
    calls = reg.all_calls()
    assert len(calls) == 2


def test_context_manager_resets():
    reg = MockToolRegistry()
    reg.register("tool", return_value="ok")
    with reg:
        reg.call("tool")
    assert reg.get("tool").call_count == 0


def test_len():
    reg = MockToolRegistry()
    reg.register("a")
    reg.register("b")
    assert len(reg) == 2


def test_contains():
    reg = MockToolRegistry()
    reg.register("tool")
    assert "tool" in reg
    assert "other" not in reg


def test_last_call():
    reg = MockToolRegistry()
    reg.register("t", return_value="ok")
    reg.call("t", x=1)
    reg.call("t", x=2)
    assert reg.get("t").last_call.args == {"x": 2}


def test_has():
    reg = MockToolRegistry()
    reg.register("tool")
    assert reg.has("tool") is True
    assert reg.has("missing") is False
