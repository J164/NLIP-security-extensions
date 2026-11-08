"""Checkout agent: deterministic pricing and (mock) order recording. No user policy.

Requests are structured JSON: ``{"action": "quote" | "commit", "items": [...]}``.

- ``quote`` prices the items from the catalog and returns a ``cart_total`` with a quote id.
- ``commit`` records an order for the given items at catalog prices.

The checkout agent never sees the user's spending limits. The orchestrator owns the user's
policy: it applies the limits from the login to the quote and decides whether to commit, ask
for confirmation, or refuse. Commit executes an order the orchestrator has already approved,
like a payment processor capturing a charge the storefront decided to make.
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
        quote_id = f"q-{token_hex(4)}"
        return protocol.reply(
            f"Quote {quote_id}: total ${total:.2f}.",
            {
                protocol.CART_TOTAL: {
                    "quote_id": quote_id,
                    "amount": total,
                    "currency": "USD",
                    "items": lines,
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
