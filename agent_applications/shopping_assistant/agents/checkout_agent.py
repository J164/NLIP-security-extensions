"""Checkout agent: deterministic quoting, spending-limit check, and (mock) order commit.

Requests are structured JSON: ``{"action": "quote" | "commit", "items": [...]}`` with the
user's scope in a ``user_scope`` submessage. Commit trusts its caller, like a payment
service trusts the storefront that holds the merchant credentials: deciding *whether* to
buy is the orchestrator's job.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

from nlip_sdk import AllowedFormat, NLIPMessage, NLIPServer, NLIPSession
from nlip_sdk.nlip_security_extensions import SecurityExtensionManager

from .. import catalog, protocol
from .._config import CONFIG, DATA_DIR
from . import serve

ENTITY_NAME = "checkout_agent"


def orders_path() -> Path:
    return Path(os.getenv("SHOP_ORDERS_PATH", DATA_DIR / "orders.jsonl"))


def price_items(items: Any) -> tuple[list[dict[str, Any]], float]:
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list")
    lines = []
    for item in items:
        product = catalog.products().get(str(item.get("product_id")))
        quantity = int(item.get("quantity", 1))
        if product is None:
            raise ValueError(f"unknown product {item.get('product_id')!r}")
        if not 1 <= quantity <= 10:
            raise ValueError("quantity must be between 1 and 10")
        lines.append(
            {
                "product_id": product["id"],
                "title": product["title"],
                "unit_price": product["price"],
                "quantity": quantity,
            }
        )
    total = round(sum(line["unit_price"] * line["quantity"] for line in lines), 2)
    return lines, total


class CheckoutAgent(NLIPServer):
    def __init__(self) -> None:
        security = SecurityExtensionManager.from_config(CONFIG, entity=ENTITY_NAME)
        super().__init__(ENTITY_NAME, security=security)

    async def handle(self, message: NLIPMessage, session: NLIPSession) -> NLIPMessage:
        if message.format != AllowedFormat.structured or not isinstance(message.content, dict):
            return protocol.reply('Send structured JSON: {"action": "quote"|"commit", "items": [...]}.')
        action = message.content.get("action")
        try:
            lines, total = price_items(message.content.get("items"))
        except (ValueError, TypeError, AttributeError) as error:
            return protocol.reply(f"Checkout error: {error}.")
        if action == "quote":
            return self._quote(message, lines, total)
        if action == "commit":
            return self._commit(message, lines, total)
        return protocol.reply(f"Unknown checkout action {action!r}.")

    def _quote(self, message: NLIPMessage, lines: list[dict], total: float) -> NLIPMessage:
        scope = dict(protocol.structured_fields(message)).get(protocol.USER_SCOPE) or {}
        max_purchase = float(scope.get("max_purchase", 0))
        auto_confirm_under = float(scope.get("auto_confirm_under", 0))
        authorized = total <= max_purchase
        required = total >= auto_confirm_under
        quote_id = f"q-{token_hex(4)}"
        if authorized:
            reason = f"${total:.2f} is within the ${max_purchase:.2f} spending limit"
        else:
            reason = f"${total:.2f} exceeds the ${max_purchase:.2f} spending limit"
        text = f"Quote {quote_id}: total ${total:.2f}. Purchase {'authorized' if authorized else 'NOT authorized'}: {reason}."
        if authorized and required:
            text += " The user must confirm this purchase."
        return protocol.reply(
            text,
            {
                protocol.CART_TOTAL: {"amount": total, "currency": "USD", "items": lines},
                protocol.PURCHASE_AUTHORIZED: {
                    "authorized": authorized,
                    "amount": total,
                    "quote_id": quote_id,
                    "reason": reason,
                },
                protocol.REQUIRES_CONFIRMATION: {
                    "required": required,
                    "threshold": auto_confirm_under,
                },
            },
        )

    def _commit(self, message: NLIPMessage, lines: list[dict], total: float) -> NLIPMessage:
        order = {
            "order_id": f"ord-{token_hex(5)}",
            "customer": message.content.get("customer"),
            "amount": total,
            "items": lines,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = orders_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write(json.dumps(order) + "\n")
        return protocol.reply(
            f"Order {order['order_id']} placed: ${total:.2f}.",
            {protocol.ORDER_ID: {k: order[k] for k in ("order_id", "amount", "items")}},
        )


nlip_agent = CheckoutAgent()
app = nlip_agent.create_nlip_app()


def main() -> None:
    serve(app, ENTITY_NAME)


if __name__ == "__main__":
    main()
