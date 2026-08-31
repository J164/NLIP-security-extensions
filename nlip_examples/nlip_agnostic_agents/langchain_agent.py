"""Minimal multi-turn chatbot built with a LangChain agent."""

import asyncio
import os
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class LangChainTestAgent:
    """A LangChain agent with process-local, in-memory conversation history."""

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str = "You are a helpful chatbot.",
        tools: Sequence[BaseTool] = (),
    ) -> None:
        # Leaving base_url unset lets ChatOpenAI use the normal OpenAI endpoint.
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before running the agent.")

        chat_model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            # LiteLLM commonly exposes the OpenAI chat-completions-compatible API.
            use_responses_api=False,
        )
        self._agent = create_agent(
            model=chat_model,
            tools=list(tools),
            system_prompt=system_prompt,
        )
        # create_agent receives full history on each invocation; keep it in this wrapper.
        self._messages: list[BaseMessage] = []

    async def chat(self, prompt: str) -> str:
        """Send one turn and retain the resulting conversation history."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        result = await self._agent.ainvoke(
            {"messages": [*self._messages, HumanMessage(content=prompt)]}
        )
        self._messages = result["messages"]
        response = self._messages[-1]
        if not isinstance(response, AIMessage):
            raise RuntimeError("LangChain agent returned no assistant message.")
        return response.text.strip()


async def main() -> None:
    print("LangChainTestAgent chatbot. Type /exit to quit.")
    agent = LangChainTestAgent()

    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            break

        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not prompt:
            continue

        print(f"Agent: {await agent.chat(prompt)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
