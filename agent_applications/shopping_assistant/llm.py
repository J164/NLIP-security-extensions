"""LangChain tool-calling agents backed by an OpenAI-compatible endpoint."""

from __future__ import annotations

import os
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
            model=os.getenv("SHOP_LLM_MODEL", settings["model"]),
            base_url=os.getenv("SHOP_LLM_BASE_URL", settings["base_url"]),
            api_key=os.getenv(settings["api_key_env"], settings["default_api_key"]),
            temperature=settings.get("temperature", 0.0),
            use_responses_api=False,
        )
        self._agent = create_agent(
            model=chat_model, tools=list(tools), system_prompt=system_prompt
        )
        self._messages: list[BaseMessage] = []

    async def chat(self, prompt: str) -> str:
        result = await self._agent.ainvoke(
            {"messages": [*self._messages, HumanMessage(content=prompt)]}
        )
        self._messages = result["messages"]
        response = self._messages[-1]
        if not isinstance(response, AIMessage):
            raise RuntimeError("Agent returned no assistant message.")
        return response.text.strip()


def model_name() -> str:
    return os.getenv("SHOP_LLM_MODEL", CONFIG["llm"]["model"])
