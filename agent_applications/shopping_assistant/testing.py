"""Scripted stand-in for ``LLMAgent``, for tests and offline reproduction without a model.

A script plays the model: it calls the real tools in a fixed order. Tools that take a
``ToolRuntime`` get one carrying this agent's state, and a returned ``Command`` is applied the
way LangGraph applies it between steps with no reducer on the key: each key in the update
overrides the stored value (LangGraph Graph API docs, State > Reducers). The script receives
the text of the tool message, as the model would.

Calls run one at a time, so this never produces LangGraph's same-step concurrent-update error;
``tests/test_shopping_app.py`` runs the real graph for that.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from itertools import count
from typing import Any

from langchain_core.tools import BaseTool
from langchain.tools import ToolRuntime
from langgraph.types import Command

Script = Callable[..., Awaitable[str]]


def _takes_runtime(tool: BaseTool) -> bool:
    function = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    return function is not None and "runtime" in inspect.signature(function).parameters


class _BoundTool:
    def __init__(self, tool: BaseTool, agent: "ScriptedAgent") -> None:
        self.tool, self.agent, self.name = tool, agent, tool.name

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        if not _takes_runtime(self.tool):
            return await self.tool.ainvoke(args)
        runtime = ToolRuntime(
            state={**self.agent.state, "messages": []},
            context=None,
            config={},
            stream_writer=lambda *_: None,
            tool_call_id=f"call-{next(self.agent.ids)}",
            store=None,
        )
        result = await self.tool.ainvoke({**args, "runtime": runtime})
        if not isinstance(result, Command):
            return result
        text = None
        for key, value in (result.update or {}).items():
            if key == "messages":
                text = value[-1].content
            else:
                self.agent.state[key] = value  # default reducer: override
        return text


class ScriptedAgent:
    """``script(tools, prompt[, turn])`` plays the model; ``tools`` maps name -> tool."""

    def __init__(self, script: Script, system_prompt: str, tools: Sequence[BaseTool], **_: Any) -> None:
        self.script, self.system_prompt = script, system_prompt
        self.state: dict[str, Any] = {}
        self.ids = count()
        self.tools = {tool.name: _BoundTool(tool, self) for tool in tools}
        self.turn = 0
        self._with_turn = len(inspect.signature(script).parameters) >= 3

    async def chat(self, prompt: str) -> str:
        self.turn += 1
        if self._with_turn:
            return await self.script(self.tools, prompt, self.turn)
        return await self.script(self.tools, prompt)


def scripted(script: Script) -> Callable[..., ScriptedAgent]:
    """An agent factory whose 'model' is ``script``."""

    def factory(system_prompt: str, tools: Sequence[BaseTool], **kwargs: Any) -> ScriptedAgent:
        return ScriptedAgent(script, system_prompt, tools, **kwargs)

    return factory
