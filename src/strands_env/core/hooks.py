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

"""Strands hooks for environment-level control of the agent tool loop."""

import logging
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import MessageAddedEvent

from .types import AgentToolStopError

logger = logging.getLogger(__name__)


class StopOnToolHook(HookProvider):
    """End the agent loop after a designated "terminal" tool is called.

    Strands ends an episode only when the model emits a turn with no tool call
    (natural stop) or a `ToolLimiter` cap is hit. A tool such as `done` that is
    *meant* to end the episode has no effect on its own — the loop continues, and
    a model that keeps re-emitting the terminal tool spins until the tool-iteration
    cap, producing a truncated, wasteful trajectory. This hook closes that gap:
    when one of `stop_tools` is called, it raises `AgentToolStopError` after the
    iteration completes, so the terminal tool's call + result are in the
    trajectory and the episode ends cleanly as `TerminationReason.TASK_COMPLETE`.

    Mirrors `ToolLimiter`'s mechanism: it arms on the terminal `toolUse` (assistant
    message) and raises on the following `toolResult` (user message), so the
    trajectory is clean without token truncation. Parallel tool calls in the same
    turn are fine — the terminal tool's result is captured before the stop.

    Example:
        >>> agent = Agent(model=model, tools=[...], hooks=[StopOnToolHook({"done"})])
        >>> try:
        ...     agent.invoke("solve this")
        ... except AgentToolStopError:
        ...     pass  # episode ended cleanly when `done` was called
    """

    def __init__(self, stop_tools: set[str] | list[str] | tuple[str, ...]):
        """Initialize a `StopOnToolHook` instance.

        Args:
            stop_tools: tool names that should end the episode when called.
        """
        self.stop_tools = set(stop_tools)
        self.reset()

    def reset(self) -> None:
        """Reset state for a new invocation."""
        self._stop_pending = False
        self.stop_tool_name: str | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks with the strands agent."""
        registry.add_callback(MessageAddedEvent, self._on_message_added)

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Arm on a terminal `toolUse`; raise once its `toolResult` arrives."""
        message = event.message
        content = message["content"]

        if message.get("role") == "assistant":
            for c in content:
                tool_use = c.get("toolUse")
                if tool_use and tool_use.get("name") in self.stop_tools:
                    self._stop_pending = True
                    self.stop_tool_name = tool_use.get("name")
                    break
        elif message.get("role") == "user":
            if self._stop_pending and any(c.get("toolResult") for c in content):
                logger.debug("Terminal tool %r called — ending agent loop", self.stop_tool_name)
                raise AgentToolStopError(f"Terminal tool {self.stop_tool_name!r} called; ending episode.")
