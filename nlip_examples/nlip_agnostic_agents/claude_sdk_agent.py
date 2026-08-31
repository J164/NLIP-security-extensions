"""Minimal multi-turn chatbot backed by the Claude Agent SDK."""

import asyncio
from shutil import which
from typing import Sequence

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import McpServerConfig


class ClaudeTestAgent:
    """A persistent Claude conversation that reuses the local Claude Code login."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful chatbot.",
        model: str = "haiku",
        mcp_servers: dict[str, McpServerConfig] | None = None,
        allowed_tools: Sequence[str] = (),
    ) -> None:
        cli_path = which("claude")
        if cli_path is None:
            raise RuntimeError("Claude Code is not installed or is not on PATH.")

        self._client = ClaudeSDKClient(
            ClaudeAgentOptions(
                cli_path=cli_path,
                system_prompt=system_prompt,
                model=model,
                # Disable Claude Code's built-in tools. Integration examples inject
                # only the explicitly allowed in-process MCP tools below.
                tools=[],
                mcp_servers=mcp_servers or {},
                allowed_tools=list(allowed_tools),
            )
        )
        self._connected = False

    async def __aenter__(self) -> "ClaudeTestAgent":
        # Reusing one connected client preserves conversation context across chat calls.
        await self._client.connect()
        self._connected = True
        return self

    async def __aexit__(self, *_: object) -> None:
        self._connected = False
        await self._client.disconnect()

    async def chat(self, prompt: str) -> str:
        """Send one turn while preserving this client's conversation context."""
        if not self._connected:
            raise RuntimeError("Use ClaudeTestAgent as an async context manager.")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        await self._client.query(prompt)
        text: list[str] = []

        # The SDK streams several event types; callers only need the assistant text.
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                text.extend(
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage) and message.is_error:
                detail = message.result or "; ".join(message.errors or []) or message.subtype
                raise RuntimeError(f"Claude request failed: {detail}")

        return "".join(text).strip()


async def main() -> None:
    print("ClaudeTestAgent chatbot. Type /exit to quit.")

    try:
        async with ClaudeTestAgent() as agent:
            while True:
                try:
                    prompt = input("You: ").strip()
                except EOFError:
                    break

                if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
                    break
                if not prompt:
                    continue

                print(f"Claude: {await agent.chat(prompt)}")
    except ClaudeSDKError as error:
        raise SystemExit(
            f"Claude Agent SDK error: {error}\n"
            "Run `claude` once and sign in with your Claude Code subscription."
        ) from error


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
