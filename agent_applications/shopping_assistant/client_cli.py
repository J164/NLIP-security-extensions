"""Scripted NLIP user client for the shopping assistant (mock login only).

    python -m agent_applications.shopping_assistant.client_cli --user alice@example.com \
        "Find noise-cancelling headphones under $150" "Buy the best rated one" --confirm

With no prompts it starts an interactive chat; type /confirm or /cancel for pending orders.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from nlip_sdk import NLIPClient, NLIPClientSession, NLIPMessage

from . import protocol
from ._config import CONFIG


def orchestrator_base() -> str:
    return CONFIG["entities"]["orchestrator"]["endpoint"].removesuffix("/nlip")


async def mock_login(base: str, email: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(f"{base}/auth/mock/token", json={"email": email})
        response.raise_for_status()
        return response.json()["ticket"]


class ShoppingClient:
    def __init__(self, base: str, ticket: str) -> None:
        self.ticket = ticket
        self.session: NLIPClientSession = NLIPClient(
            {"orchestrator": f"{base}/nlip"}, identity="cli_user"
        ).create_session("orchestrator")

    async def send(self, message: NLIPMessage) -> NLIPMessage:
        message.add_authentication_token(self.ticket)
        return await self.session.send(message)

    async def say(self, text: str) -> NLIPMessage:
        return await self.send(NLIPMessage.text(text))

    async def confirm(self, confirm: bool = True) -> NLIPMessage:
        action = "confirm_order" if confirm else "cancel_order"
        return await self.send(NLIPMessage.structured({"action": action}))


def pending_order(response: NLIPMessage) -> dict | None:
    status = dict(protocol.structured_fields(response)).get(protocol.ORDER_STATUS) or {}
    return status.get("pending_order")


def show(response: NLIPMessage, verbose: bool) -> None:
    print(f"assistant: {response.content}")
    fields = dict(protocol.structured_fields(response))
    for step in fields.get(protocol.TRACE) or []:
        print(f"  trace: {json.dumps(step)}")
    status = fields.get(protocol.ORDER_STATUS) or {}
    if status.get("pending_order"):
        print(f"  pending order: ${status['pending_order']['amount']} (confirm with /confirm)")
    if status.get("last_order"):
        print(f"  last order: {status['last_order']['order_id']}")
    if verbose:
        print(json.dumps(response.to_dict(), indent=2))


async def run(args: argparse.Namespace) -> None:
    client = ShoppingClient(args.base, await mock_login(args.base, args.user))
    if args.prompts:
        response = None
        for prompt in args.prompts:
            print(f"user: {prompt}")
            response = await client.say(prompt)
            show(response, args.verbose)
        if args.confirm and response is not None and pending_order(response):
            print("user: [confirm]")
            show(await client.confirm(), args.verbose)
        return
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            break
        if prompt in {"/exit", "/quit"}:
            break
        if prompt in {"/confirm", "/cancel"}:
            show(await client.confirm(prompt == "/confirm"), args.verbose)
        elif prompt:
            show(await client.say(prompt), args.verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompts", nargs="*", help="Prompts to send in order.")
    parser.add_argument("--user", default="alice@example.com", help="Mock identity to log in as.")
    parser.add_argument("--base", default=orchestrator_base(), help="Orchestrator base URL.")
    parser.add_argument("--confirm", action="store_true", help="Confirm a pending order at the end.")
    parser.add_argument("--verbose", action="store_true", help="Print full NLIP responses.")
    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
