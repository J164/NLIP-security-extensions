"""The predicate vocabulary the AgentSpec rules are written in.

Two kinds of context:

- ``PartContext``: one part of a reply that a subagent sent to the orchestrator, checked
  at the orchestrator's client-side ingress, before the reply is merged into order state
  or flattened into the model's prompt (trigger ``before_flatten``).
- ``ActionContext``: one tool call the orchestrator's model wants to make (trigger
  ``before_action``).

The sender of a part is ``SecurityEvent.peer``: the endpoint the orchestrator itself
dialed, not anything the message claims about itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nlip_sdk import AllowedFormat, NLIPSubMessage

# Which entity may emit each reserved field label (draft plan section 3).
DEFAULT_FIELD_OWNERS = {
    "product_results": "search_agent",
    "review_summary": "review_agent",
    "cart_total": "checkout_agent",
    "purchase_authorized": "checkout_agent",
    "requires_confirmation": "checkout_agent",
    "order_id": "checkout_agent",
    "user_scope": "orchestrator",
    "order_status": "orchestrator",
    "trace": "orchestrator",
    "provenance": "orchestrator",  # the tag this SE adds; a subagent may not forge one
}
DEFAULT_AGENTS = ("orchestrator", "search_agent", "review_agent", "checkout_agent")
# Chat roles a label could claim; agent names are added from ``agents``.
DEFAULT_ROLES = ("system", "developer", "assistant", "user", "tool", "function")
DEFAULT_LABEL_PATTERN = r"[A-Za-z][A-Za-z0-9_.\-]{0,63}"
# A line in the orchestrator's prompt rendering starts with "[from <entity>]".
PROVENANCE_LINE = re.compile(r"\n\s*\[\s*from\s+[^\]\n]+\]", re.IGNORECASE)


@dataclass(frozen=True)
class Policy:
    field_owners: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_FIELD_OWNERS))
    privileged_labels: frozenset[str] = frozenset(DEFAULT_ROLES + DEFAULT_AGENTS)
    agents: frozenset[str] = frozenset(DEFAULT_AGENTS)
    orchestrator: str = "orchestrator"
    label_pattern: re.Pattern[str] = re.compile(DEFAULT_LABEL_PATTERN)

    @classmethod
    def from_options(
        cls,
        field_owners: Mapping[str, str] | None = None,
        roles: list[str] | None = None,
        agents: list[str] | None = None,
        orchestrator: str = "orchestrator",
        label_pattern: str = DEFAULT_LABEL_PATTERN,
    ) -> "Policy":
        agents = [a.casefold() for a in (agents or DEFAULT_AGENTS)]
        roles = [r.casefold() for r in (roles or DEFAULT_ROLES)]
        owners = {k.casefold(): v.casefold() for k, v in (field_owners or DEFAULT_FIELD_OWNERS).items()}
        return cls(owners, frozenset(roles + agents), frozenset(agents), orchestrator.casefold(),
                   re.compile(label_pattern))


def _norm(label: str | None) -> str:
    return (label or "").strip().casefold()


# --- part predicates (trigger before_flatten) ----------------------------------------------

@dataclass(frozen=True)
class PartContext:
    sender: str | None  # the peer the orchestrator dialed; None for user traffic
    part: NLIPSubMessage
    index: int  # 0 is the message's primary part
    policy: Policy


def provenance(ctx: PartContext) -> str | None:
    return ctx.sender.casefold() if ctx.sender else None


def field_owner(ctx: PartContext) -> str | None:
    return ctx.policy.field_owners.get(_norm(ctx.part.label))


def from_subagent(ctx: PartContext) -> bool:
    """The part arrived on a reply from a registered agent other than the orchestrator."""
    sender = provenance(ctx)
    return sender is not None and sender in ctx.policy.agents and sender != ctx.policy.orchestrator


def is_token(ctx: PartContext) -> bool:
    """Protocol tokens (conversation ids) are transport metadata, never shown or merged."""
    return ctx.part.format == AllowedFormat.token


def has_label(ctx: PartContext) -> bool:
    return ctx.part.label is not None


def is_privileged_label(ctx: PartContext) -> bool:
    """The label names a chat role or an agent, exactly (``user_rating`` is not ``user``)."""
    return _norm(ctx.part.label) in ctx.policy.privileged_labels


def is_wellformed_label(ctx: PartContext) -> bool:
    """A short identifier, so it cannot carry line breaks or forged prompt lines."""
    return ctx.part.label is None or bool(ctx.policy.label_pattern.fullmatch(ctx.part.label))


def is_reserved_field(ctx: PartContext) -> bool:
    return field_owner(ctx) is not None


def sender_owns_field(ctx: PartContext) -> bool:
    owner = field_owner(ctx)
    return owner is not None and owner == provenance(ctx)


def forges_provenance(ctx: PartContext) -> bool:
    """String content with a line that would render like another entity's prompt line."""
    return isinstance(ctx.part.content, str) and bool(PROVENANCE_LINE.search(ctx.part.content))


# --- action predicates (trigger before_action) ---------------------------------------------

@dataclass(frozen=True)
class ActionContext:
    tool: str
    args: Mapping[str, Any]
    user_scope: Mapping[str, Any]  # from the OAuth ticket, never from a subagent
    state: Mapping[str, Any]


def _amount(ctx: ActionContext) -> float | None:
    value = ctx.args.get("amount")
    return float(value) if isinstance(value, (int, float)) else None


def user_scope(ctx: ActionContext) -> Mapping[str, Any]:
    return ctx.user_scope


def has_amount(ctx: ActionContext) -> bool:
    return _amount(ctx) is not None


def exceeds_max_purchase(ctx: ActionContext) -> bool:
    amount, limit = _amount(ctx), user_scope(ctx).get("max_purchase")
    return amount is not None and limit is not None and amount > float(limit)


def at_or_above_confirm_threshold(ctx: ActionContext) -> bool:
    amount, threshold = _amount(ctx), user_scope(ctx).get("auto_confirm_under")
    return amount is not None and threshold is not None and amount >= float(threshold)


def purchase_authorized(ctx: ActionContext) -> bool:
    """Order state says checkout authorized the purchase (same test as ``OrderState``)."""
    field = ctx.state.get("purchase_authorized")
    return isinstance(field, Mapping) and bool(field.get("authorized"))


def confirmation_waived(ctx: ActionContext) -> bool:
    """Order state says no confirmation is needed (same test as ``OrderState``)."""
    field = ctx.state.get("requires_confirmation")
    return isinstance(field, Mapping) and bool(field) and not field.get("required", True)


PREDICATES = {
    fn.__name__: fn
    for fn in (
        from_subagent, is_token, has_label, is_privileged_label, is_wellformed_label,
        is_reserved_field, sender_owns_field, forges_provenance,
        has_amount, exceeds_max_purchase, purchase_authorized, at_or_above_confirm_threshold, confirmation_waived,
    )
}
