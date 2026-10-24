"""LangChain tool-calling agents backed by an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from ._config import CONFIG


class ChatAgent(Protocol):
    async def chat(self, prompt: str) -> str: ...


# Signature used to build an agent; tests substitute a scripted one.
AgentFactory = Callable[[str, Sequence[BaseTool]], ChatAgent]


class LLMAgent:
    """A multi-turn tool-calling agent that keeps its conversation in memory.

    Same shape as ``nlip_examples``' ``LangChainTestAgent``, plus a configurable
    temperature so evaluation runs are as repeatable as the model allows.
    """

    def __init__(self, system_prompt: str, tools: Sequence[BaseTool] = ()) -> None:
        settings = CONFIG["llm"]
        chat_model = ChatOpenAI(
            model=model_name(),
            base_url=os.getenv("SHOP_LLM_BASE_URL", settings["base_url"]),
            api_key=os.getenv(settings["api_key_env"], settings["default_api_key"]),
            temperature=settings.get("temperature", 0.0),
            max_tokens=settings.get("max_tokens"),
            timeout=settings.get("timeout_seconds"),
            use_responses_api=False,
        )
        self._agent = create_agent(
            model=chat_model, tools=list(tools), system_prompt=system_prompt
        )
        self._messages: list[BaseMessage] = []

    async def chat(self, prompt: str) -> str:
        started = time.monotonic()
        seen = len(self._messages)
        result = await self._agent.ainvoke(
            {"messages": [*self._messages, HumanMessage(content=prompt)]}
        )
        self._messages = result["messages"]
        log_usage(self._messages[seen:], time.monotonic() - started)
        response = self._messages[-1]
        if not isinstance(response, AIMessage):
            raise RuntimeError("Agent returned no assistant message.")
        return response.text.strip()


def log_usage(messages: Sequence[BaseMessage], latency_s: float) -> None:
    """Append token usage for one chat() call to $SHOP_LLM_USAGE_LOG, if set."""
    path = os.getenv("SHOP_LLM_USAGE_LOG")
    if not path:
        return
    replies = [m for m in messages if isinstance(m, AIMessage)]
    usage = [m.usage_metadata or {} for m in replies]
    row = {
        "ts": time.time(),
        "entity": os.getenv("SHOP_ENTITY", ""),
        "model": model_name(),
        "llm_calls": len(replies),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "latency_s": round(latency_s, 3),
    }
    with open(path, "a") as file:
        file.write(json.dumps(row) + "\n")


def model_name() -> str:
    return os.getenv("SHOP_LLM_MODEL", CONFIG["llm"]["model"])
