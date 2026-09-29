"""Per-conversation order state kept by the orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from nlip_sdk import NLIPMessage

from . import protocol


@dataclass
class OrderState:
    """What the orchestrator believes about the current order.

    ``purchase_authorized`` and ``requires_confirmation`` gate the privileged
    ``place_order`` action. They are meant to come from the checkout agent.
    """

    user_scope: dict[str, Any] = field(default_factory=dict)
    product_results: dict[str, Any] | None = None
    review_summary: dict[str, Any] | None = None
    cart_total: dict[str, Any] | None = None
    purchase_authorized: dict[str, Any] | None = None
    requires_confirmation: dict[str, Any] | None = None
    pending_order: dict[str, Any] | None = None
    orders: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    def is_authorized(self) -> bool:
        return bool(self.purchase_authorized and self.purchase_authorized.get("authorized"))

    def needs_confirmation(self) -> bool:
        # Missing information is treated as "ask the user".
        return not self.requires_confirmation or bool(
            self.requires_confirmation.get("required", True)
        )

    def reset_quote(self) -> None:
        self.cart_total = None
        self.purchase_authorized = None
        self.requires_confirmation = None
        self.pending_order = None


# Labels whose content is copied into OrderState.
MERGED_FIELDS = {
    protocol.PRODUCT_RESULTS,
    protocol.REVIEW_SUMMARY,
    protocol.CART_TOTAL,
    protocol.PURCHASE_AUTHORIZED,
    protocol.REQUIRES_CONFIRMATION,
    protocol.USER_SCOPE,
}


def merge_labeled_fields(state: OrderState, message: NLIPMessage) -> list[str]:
    """Copy every known labeled structured field into the order state.

    This is the ordinary "shared blackboard" pattern: the label names the field and the
    last writer wins. It does not ask *who* sent the field, because NLIP gives the
    orchestrator no per-submessage provenance. It is deliberately left unguarded here;
    enforcing field ownership is the job of the security extension.
    """
    merged = []
    for label, content in protocol.structured_fields(message):
        if label in MERGED_FIELDS and isinstance(content, dict):
            setattr(state, label, content)
            merged.append(label)
        elif label == protocol.ORDER_ID and isinstance(content, dict):
            state.orders.append(content)
            merged.append(label)
    return merged
