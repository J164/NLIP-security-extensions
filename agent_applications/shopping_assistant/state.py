"""Order state, kept in the orchestrator agent's LangGraph state.

The order fields are extra keys on the agent's LangGraph state, declared through middleware
(the form the LangChain v1 migration guide prefers over ``create_agent(state_schema=...)``).
None of them has a reducer, so LangGraph's default applies: "If no reducer function is
explicitly specified then it is assumed that all updates to that key should override it"
(LangGraph Graph API docs, State > Reducers).

The orchestrator decides on the user's policy from ``cart_total`` (see ``policy.py``), so the
quote is the one field that matters for spending.

A tool that talks to a subagent returns ``Command(update=...)`` with the labeled fields of
the reply (LangChain Tools docs, "Updating State with Command"). So the label names the key
and the last writer wins. Nothing asks *who* sent the field, because NLIP gives the
orchestrator no per-submessage provenance. That is the weakness the security extension is
meant to close; it is deliberately left in the app.

LangGraph does refuse two writes to the same key in one step (INVALID_CONCURRENT_GRAPH_UPDATE),
for example two parallel tool calls that both return ``cart_total``. Writes in later steps
override silently.

State the orchestrator owns itself (the user's scope from the login, the pending order the
user is asked to confirm, and the order history) is not in the graph state: no subagent
reply can reach it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState

from nlip_sdk import NLIPMessage

from . import protocol

# Labels whose content becomes a graph-state update when a subagent reply arrives.
# ``user_scope`` is not one of them: only the orchestrator sets it, from the OAuth login.
ORDER_FIELDS = (protocol.PRODUCT_RESULTS, protocol.REVIEW_SUMMARY, protocol.CART_TOTAL)
# The checkout agent's quote; cleared before a new quote and after a commit.
QUOTE_FIELDS = (protocol.CART_TOTAL,)


class OrderState(AgentState):
    """The agent state plus the order fields. No reducers: every update overrides the key."""

    product_results: NotRequired[dict[str, Any] | None]
    review_summary: NotRequired[dict[str, Any] | None]
    cart_total: NotRequired[dict[str, Any] | None]


class OrderStateMiddleware(AgentMiddleware[OrderState]):
    """Adds the order fields to the orchestrator agent's state."""

    state_schema = OrderState


def merged_labels(message: NLIPMessage) -> list[str]:
    """The labels of a reply that update order state or the order history, in message order."""
    return [
        label
        for label, content in protocol.structured_fields(message)
        if isinstance(content, dict) and (label in ORDER_FIELDS or label == protocol.ORDER_ID)
    ]


def labeled_update(message: NLIPMessage) -> dict[str, Any]:
    """The graph-state update a reply produces: each order-field label names a key.

    A label repeated within one reply keeps its last value, as consecutive writes would.
    """
    return {
        label: content
        for label, content in protocol.structured_fields(message)
        if label in ORDER_FIELDS and isinstance(content, dict)
    }


def order_ids(message: NLIPMessage) -> list[dict[str, Any]]:
    """Order confirmations in a reply, for the orchestrator's order history."""
    return [
        content
        for label, content in protocol.structured_fields(message)
        if label == protocol.ORDER_ID and isinstance(content, dict)
    ]


def cleared_quote() -> dict[str, None]:
    return {field: None for field in QUOTE_FIELDS}


def snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    return {field: state.get(field) for field in ORDER_FIELDS}

