"""Interactive user client that can talk to any configured NLIP agent."""

from __future__ import annotations

import asyncio
from typing import Any

from nlip_examples.nlip_integration_examples._config import CONFIG
from nlip_sdk import NLIPClient, NLIPClientSession


class UserChatClient:
    """A non-agent caller that selects one destination for each NLIP exchange."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or CONFIG
        # One client owns the complete destination directory; no per-agent clients.
        self.client = NLIPClient.from_config(config, identity="user")
        self.sessions: dict[str, NLIPClientSession] = {
            destination: self.client.create_session(destination)
            for destination in self.client.destinations
        }
        self.target = str(config.get("user", {}).get("default_target", "")).casefold()
        if self.target not in self.client.destinations:
            self.target = self.client.destinations[0]

    def use(self, target: str) -> None:
        target = target.casefold()
        if target not in self.client.destinations:
            raise ValueError(f"Unknown agent {target!r}.")
        self.target = target

    def new_session(self, target: str | None = None) -> NLIPClientSession:
        """Start a fresh conversation without affecting other destinations."""
        target = (target or self.target).casefold()
        if target not in self.client.destinations:
            raise ValueError(f"Unknown agent {target!r}.")
        self.sessions[target] = self.client.create_session(target)
        return self.sessions[target]

    async def chat(self, prompt: str, target: str | None = None) -> str:
        target = (target or self.target).casefold()
        if target not in self.client.destinations:
            raise ValueError(f"Unknown agent {target!r}.")
        response = await self.sessions[target].send_text(prompt)
        return response.extract_text(language=None) or response.to_json()


async def main() -> None:
    client = UserChatClient()
    agents = ", ".join(client.client.destinations)
    print(
        f"NLIP user client. Agents: {agents}. "
        "Commands: /use NAME, /new, /agents, /exit"
    )

    while True:
        try:
            prompt = input(f"You -> {client.target}: ").strip()
        except EOFError:
            break

        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if prompt == "/agents":
            print(f"Agents: {agents}")
            continue
        if prompt == "/new":
            session = client.new_session()
            print(f"Started a new {client.target} session: {session.id}")
            continue
        if prompt.startswith("/use "):
            try:
                client.use(prompt.removeprefix("/use ").strip())
            except ValueError as error:
                print(error)
            continue
        if not prompt:
            continue

        try:
            print(f"{client.target}: {await client.chat(prompt)}")
        except Exception as error:
            print(f"NLIP request failed: {error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
