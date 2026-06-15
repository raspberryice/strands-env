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

from types import SimpleNamespace

import pytest

from strands_env.core.hooks import StopOnToolHook
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
