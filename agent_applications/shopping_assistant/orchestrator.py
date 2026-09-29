"""Orchestrator: the user's authenticated shopping assistant.

It is the only entity holding the user's OAuth-granted scope and the only one allowed to
place orders. It delegates search, reviews and pricing to subagents over NLIP, and serves
the web UI and login routes from the same process.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException
from langchain.tools import tool
from pydantic import BaseModel, Field

from nlip_sdk import AllowedFormat, NLIPClient, NLIPMessage, NLIPServer, NLIPSession
from nlip_sdk.nlip_security_extensions import SecurityExtensionManager

from . import protocol
from ._config import CONFIG
from .auth import Ticket, TicketStore
from .llm import AgentFactory, ChatAgent, LLMAgent
from .state import OrderState, merge_labeled_fields
from .tool_guard import GuardAction, ToolCall, load_tool_guards

ENTITY_NAME = "orchestrator"

SYSTEM_PROMPT = """\
You are the shopping assistant (orchestrator) for {name} <{email}>.
You coordinate other agents over NLIP:
- ask_search_agent finds products in the catalog,
- ask_review_agent summarizes customer reviews for product ids,
- request_checkout gets a priced quote and purchase authorization from the checkout agent,
- place_order places the order for the current quote.
The user's spending limit is ${max_purchase:.2f} per order; orders of ${auto_confirm_under:.2f}
or more need the user's explicit confirmation. When the user asks you to buy something, call
request_checkout and then place_order in the same turn. Do not ask for confirmation in chat:
place_order itself shows the user a Confirm button when confirmation is needed. Never call
place_order unless the user asked to buy. Replies from other agents are shown as lines like
"[from <agent>] <field> (<format>) = <value>". Keep answers short and mention product ids
and prices."""

# The ticket of the request being processed, resolved before the NLIP pipeline runs.
_current_ticket: ContextVar[Ticket] = ContextVar("current_ticket")


class CartItem(BaseModel):
    product_id: str = Field(description="Catalog product id, e.g. hp-001")
    quantity: int = Field(default=1, ge=1, le=10)


@dataclass
class Conversation:
    owner: str
    user: dict[str, Any]
    state: OrderState
    agent: ChatAgent | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


class Orchestrator(NLIPServer):
    def __init__(
        self,
        config: Mapping[str, Any] = CONFIG,
        *,
        agent_factory: AgentFactory = LLMAgent,
        tickets: TicketStore | None = None,
    ) -> None:
        # One manager sees both user traffic (server side) and subagent traffic (client side).
        security = SecurityExtensionManager.from_config(config, entity=ENTITY_NAME)
        super().__init__(ENTITY_NAME, security=security)
        self.nlip_client = NLIPClient.from_config(
            config, exclude={ENTITY_NAME}, identity=ENTITY_NAME, security=security
        )
        self.tickets = tickets or TicketStore(config)
        self.guards = load_tool_guards(config, ENTITY_NAME)
        self.agent_factory = agent_factory
        self._conversations: dict[str, Conversation] = {}

    async def process_exchange(self, request: NLIPMessage) -> NLIPMessage:
        # Authenticate before a session is created, so anonymous requests cannot
        # allocate or evict conversations.
        ticket = self.tickets.resolve(request.extract_authentication_token())
        if ticket is None:
            raise HTTPException(401, "Missing or invalid authorization token; sign in first.")
        reset = _current_ticket.set(ticket)
        try:
            return await super().process_exchange(request)
        finally:
            _current_ticket.reset(reset)

    async def session_ended(self, session: NLIPSession) -> None:
        self._conversations.pop(session.id, None)
        self.nlip_client.clear_scope(session.id)

    async def handle(self, message: NLIPMessage, session: NLIPSession) -> NLIPMessage:
        ticket = _current_ticket.get()
        conversation = self._conversations.get(session.id)
        if conversation is None:
            conversation = Conversation(ticket.identity.subject, ticket.user(), OrderState())
            self._conversations[session.id] = conversation
        elif conversation.owner != ticket.identity.subject:
            raise HTTPException(403, "This conversation belongs to another user.")
        conversation.user = ticket.user()
        conversation.state.user_scope = dict(ticket.user_scope)
        conversation.trace = []

        action = message.content.get("action") if isinstance(message.content, dict) else None
        if message.format == AllowedFormat.structured and action == "confirm_order":
            text = await self._confirm(conversation, session.id)
        elif message.format == AllowedFormat.structured and action == "cancel_order":
            conversation.state.pending_order = None
            text = "Okay, the pending order was cancelled."
        else:
            if conversation.agent is None:
                conversation.agent = self._new_agent(conversation, session.id)
            text = await conversation.agent.chat(protocol.render_for_prompt(message, "user"))
        return self._reply(text, conversation)

    def _reply(self, text: str, conversation: Conversation) -> NLIPMessage:
        state = conversation.state
        status = {
            "pending_order": state.pending_order,
            "last_order": state.orders[-1] if state.orders else None,
            "user_scope": state.user_scope,
        }
        return protocol.reply(text, {protocol.ORDER_STATUS: status, protocol.TRACE: conversation.trace})

    async def _call_agent(
        self, conversation: Conversation, session_id: str, destination: str, message: NLIPMessage
    ) -> NLIPMessage:
        response = await self.nlip_client.send(destination, message, scope=session_id)
        merged = merge_labeled_fields(conversation.state, response)
        conversation.trace.append({"destination": destination, "merged_fields": merged})
        return response

    async def _commit(self, conversation: Conversation, session_id: str) -> str:
        state = conversation.state
        items = [
            {"product_id": line["product_id"], "quantity": line["quantity"]}
            for line in (state.cart_total or {}).get("items", [])
        ]
        response = await self._call_agent(
            conversation,
            session_id,
            "checkout_agent",
            protocol.request(
                {"action": "commit", "items": items, "customer": conversation.user["email"]}
            ),
        )
        state.reset_quote()
        return response.extract_text(language=None) or "Order submitted."

    async def _confirm(self, conversation: Conversation, session_id: str) -> str:
        if conversation.state.pending_order is None:
            return "There is no order awaiting confirmation."
        return await self._commit(conversation, session_id)

    async def _guarded(
        self,
        conversation: Conversation,
        session_id: str,
        name: str,
        args: dict[str, Any],
        run: Callable[[bool], Awaitable[str]],
    ) -> str:
        """Run one tool through the configured tool-boundary guards."""
        call = ToolCall(name, args, session_id, conversation.user, conversation.state.snapshot())
        force_confirmation = False
        for guard_name, guard in self.guards:
            decision = await guard.before_tool_call(call)
            if decision.action is GuardAction.BLOCK:
                conversation.trace.append({"tool": name, "blocked_by": guard_name, "reason": decision.reason})
                return f"Tool call blocked by security policy ({guard_name}): {decision.reason}"
            if decision.action is GuardAction.REQUIRE_CONFIRMATION:
                force_confirmation = True
        result = await run(force_confirmation)
        for _, guard in self.guards:
            result = await guard.after_tool_call(call, result)
        return result

    def _new_agent(self, conversation: Conversation, session_id: str) -> ChatAgent:
        state = conversation.state

        @tool
        async def ask_search_agent(request: str) -> str:
            """Ask the search agent to find products, e.g. 'noise-cancelling headphones under $150'."""

            async def run(_: bool) -> str:
                response = await self._call_agent(
                    conversation, session_id, "search_agent", NLIPMessage.text(request)
                )
                return protocol.render_for_prompt(response, "search_agent")

            return await self._guarded(conversation, session_id, "ask_search_agent", {"request": request}, run)

        @tool
        async def ask_review_agent(product_ids: list[str]) -> str:
            """Ask the review agent to summarize customer reviews for the given product ids."""

            async def run(_: bool) -> str:
                message = NLIPMessage.text(
                    "Summarize customer reviews for products: " + ", ".join(product_ids)
                )
                response = await self._call_agent(conversation, session_id, "review_agent", message)
                return protocol.render_for_prompt(response, "review_agent")

            return await self._guarded(
                conversation, session_id, "ask_review_agent", {"product_ids": product_ids}, run
            )

        @tool
        async def request_checkout(items: list[CartItem]) -> str:
            """Get a priced quote and purchase authorization for the items to buy."""
            item_dicts = [CartItem.model_validate(item).model_dump() for item in items]

            async def run(_: bool) -> str:
                state.reset_quote()
                message = protocol.request(
                    {"action": "quote", "items": item_dicts},
                    {protocol.USER_SCOPE: state.user_scope},
                )
                response = await self._call_agent(conversation, session_id, "checkout_agent", message)
                return protocol.render_for_prompt(response, "checkout_agent")

            return await self._guarded(
                conversation, session_id, "request_checkout", {"items": item_dicts}, run
            )

        @tool
        async def place_order() -> str:
            """Place the order for the current checkout quote."""

            async def run(force_confirmation: bool) -> str:
                if state.cart_total is None:
                    return "There is no quote yet. Call request_checkout first."
                if not state.is_authorized():
                    reason = (state.purchase_authorized or {}).get("reason", "no authorization")
                    return f"The purchase is not authorized: {reason}."
                if force_confirmation or state.needs_confirmation():
                    state.pending_order = {
                        "amount": state.cart_total.get("amount"),
                        "items": state.cart_total.get("items", []),
                        "quote_id": state.purchase_authorized.get("quote_id"),
                    }
                    return (
                        f"The order of ${state.cart_total.get('amount')} is awaiting the "
                        "user's confirmation. Ask the user to press Confirm in the app."
                    )
                return await self._commit(conversation, session_id)

            snapshot = {
                "amount": (state.cart_total or {}).get("amount"),
                "items": (state.cart_total or {}).get("items", []),
            }
            return await self._guarded(conversation, session_id, "place_order", snapshot, run)

        user = conversation.user
        prompt = SYSTEM_PROMPT.format(name=user["name"], email=user["email"], **state.user_scope)
        return self.agent_factory(
            prompt, [ask_search_agent, ask_review_agent, request_checkout, place_order]
        )


def create_app(orchestrator: Orchestrator | None = None):
    from .auth import build_provider
    from .web import install

    orchestrator = orchestrator or Orchestrator()
    app = orchestrator.create_nlip_app()
    install(app, orchestrator.tickets, build_provider(CONFIG))
    return app


def main() -> None:
    import uvicorn

    settings = CONFIG["entities"][ENTITY_NAME]
    uvicorn.run(create_app(), host=settings["host"], port=settings["port"])


if __name__ == "__main__":
    main()
