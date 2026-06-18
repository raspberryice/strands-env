# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for environment control hooks."""

import logging
from types import SimpleNamespace

import pytest

from strands_env.core.hooks import _STRANDS_EVENT_LOOP_LOGGER, StopOnToolHook
from strands_env.core.types import AgentToolStopError


def _event(role: str, content: list[dict]) -> SimpleNamespace:
    """Minimal stand-in for a strands MessageAddedEvent (only `.message` is read)."""
    return SimpleNamespace(message={"role": role, "content": content})


class TestStopOnToolHook:
    def test_raises_after_terminal_tool_result(self):
        hook = StopOnToolHook({"done"})
        hook._on_message_added(_event("user", [{"text": "solve this"}]))  # task prompt
        hook._on_message_added(_event("assistant", [{"toolUse": {"name": "done"}}]))
        assert hook._stop_pending is True
        with pytest.raises(AgentToolStopError):
            hook._on_message_added(_event("user", [{"toolResult": {"content": []}}]))

    def test_ignores_non_terminal_tools(self):
        hook = StopOnToolHook({"done"})
        hook._on_message_added(_event("assistant", [{"toolUse": {"name": "submit_solution"}}]))
        assert hook._stop_pending is False
        # toolResult for a non-terminal tool must not stop the loop.
        hook._on_message_added(_event("user", [{"toolResult": {"content": []}}]))

    def test_does_not_raise_until_tool_result(self):
        # Arming on the assistant toolUse alone must not raise — the terminal
        # tool's result has to land in the trajectory first (clean stop).
        hook = StopOnToolHook({"done"})
        hook._on_message_added(_event("assistant", [{"toolUse": {"name": "done"}}]))
        # No exception yet; a plain user message (no toolResult) is a no-op.
        hook._on_message_added(_event("user", [{"text": "irrelevant"}]))

    def test_terminal_tool_alongside_other_calls(self):
        # Parallel calls: `done` in the same turn as another tool still stops.
        hook = StopOnToolHook({"done"})
        hook._on_message_added(
            _event("assistant", [{"toolUse": {"name": "submit_solution"}}, {"toolUse": {"name": "done"}}])
        )
        assert hook._stop_pending is True
        with pytest.raises(AgentToolStopError):
            hook._on_message_added(_event("user", [{"toolResult": {"content": []}}]))

    def test_reset_clears_state(self):
        hook = StopOnToolHook({"done"})
        hook._on_message_added(_event("assistant", [{"toolUse": {"name": "done"}}]))
        hook.reset()
        assert hook._stop_pending is False
        assert hook.stop_tool_name is None


class TestStopLogFilter:
    """Constructing the hook installs a filter that drops strands' "cycle failed"
    traceback for AgentToolStopError (strands logs every cycle exception), while
    leaving real errors logged."""

    def _capture(self):
        records: list[logging.LogRecord] = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record)

        el = logging.getLogger(_STRANDS_EVENT_LOOP_LOGGER)
        handler = _Cap()
        el.addHandler(handler)
        el.setLevel(logging.DEBUG)
        prev_propagate = el.propagate
        el.propagate = False
        StopOnToolHook({"done"})  # installs the suppression filter (idempotent)
        return el, handler, records, prev_propagate

    def test_suppresses_agent_tool_stop_traceback(self):
        el, handler, records, prev = self._capture()
        try:
            try:
                raise AgentToolStopError("done")
            except Exception:
                el.exception("cycle failed")  # bare
            try:
                raise AgentToolStopError("done")
            except Exception as inner:
                raise RuntimeError("wrapper") from inner
        except Exception:
            el.exception("cycle failed")  # wrapped via __cause__
        finally:
            keep = list(records)
            el.removeHandler(handler)
            el.propagate = prev
        assert keep == [], "AgentToolStopError 'cycle failed' records should be dropped"

    def test_keeps_unrelated_errors(self):
        el, handler, records, prev = self._capture()
        try:
            raise ValueError("real boom")
        except Exception:
            el.exception("cycle failed")
        finally:
            keep = list(records)
            el.removeHandler(handler)
            el.propagate = prev
        assert len(keep) == 1 and isinstance(keep[0].exc_info[1], ValueError)
