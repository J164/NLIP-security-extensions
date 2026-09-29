"""Application protocol shared by the shopping agents.

Every agent reply is one NLIP message: a text summary as the first submessage plus labeled
``structured/json`` submessages carrying machine-readable fields. The labels below are the
application's field vocabulary. ECMA-430 lets any sender put any string in ``Label``; nothing
in the protocol says which entity may emit which label.
"""

from __future__ import annotations

import base64
import binascii
import json

from nlip_sdk import AllowedFormat, NLIPMessage, NLIPSubMessage

# Field labels and the entity that legitimately produces each one.
PRODUCT_RESULTS = "product_results"  # search_agent
REVIEW_SUMMARY = "review_summary"  # review_agent
CART_TOTAL = "cart_total"  # checkout_agent
PURCHASE_AUTHORIZED = "purchase_authorized"  # checkout_agent
REQUIRES_CONFIRMATION = "requires_confirmation"  # checkout_agent
ORDER_ID = "order_id"  # checkout_agent
USER_SCOPE = "user_scope"  # orchestrator, from the OAuth login
ORDER_STATUS = "order_status"  # orchestrator -> user client
TRACE = "trace"  # orchestrator -> user client


def labeled(label: str, content: object) -> NLIPSubMessage:
    """A labeled structured JSON submessage."""
    return NLIPSubMessage(
        format=AllowedFormat.structured, subformat="json", content=content, label=label
    )


def reply(text: str, fields: dict[str, object] | None = None) -> NLIPMessage:
    """Build an agent reply: a text summary plus one labeled submessage per field."""
    message = NLIPMessage.text(text)
    for label, content in (fields or {}).items():
        message.add_submessage(labeled(label, content))
    return message


def request(content: object, fields: dict[str, object] | None = None) -> NLIPMessage:
    """Build a structured request plus labeled context submessages."""
    message = NLIPMessage.structured(content)
    for label, value in (fields or {}).items():
        message.add_submessage(labeled(label, value))
    return message


def structured_fields(message: NLIPMessage) -> list[tuple[str, object]]:
    """Every labeled structured part, in message order."""
    return [
        (part.label, part.content)
        for part in message.submessages or []
        if part.format == AllowedFormat.structured and part.label
    ]


def _render_content(part: NLIPSubMessage) -> str:
    if part.format == AllowedFormat.binary:
        try:
            size = len(base64.b64decode(str(part.content), validate=True))
        except (binascii.Error, ValueError):
            size = len(str(part.content))
        return f"<binary {part.subformat}, {size} bytes>"
    if isinstance(part.content, str):
        return part.content
    return json.dumps(part.content, ensure_ascii=False, sort_keys=True)


def render_for_prompt(message: NLIPMessage, source: str) -> str:
    """Flatten a whole NLIP message into prompt text, keeping labels and structure.

    ECMA-430 describes ``Label`` as carrying, e.g., role information in an LLM chat, so a
    faithful consumer shows the model every labeled part, not only the plain text. Token
    parts are protocol metadata (credentials, conversation ids) and are never shown.
    """
    lines = []
    for index, part in enumerate(message.iter_parts()):
        if part.format == AllowedFormat.token:
            continue
        name = part.label or ("message" if index == 0 else f"part{index}")
        kind = part.format.value
        if part.format != AllowedFormat.text:
            kind = f"{kind}/{part.subformat}"
        lines.append(f"[from {source}] {name} ({kind}) = {_render_content(part)}")
    return "\n".join(lines)
