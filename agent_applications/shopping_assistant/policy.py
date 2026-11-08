"""The user's purchase policy, applied by the orchestrator.

The orchestrator is the only entity that holds the user's scope (granted at login), so it is
the one that decides. The decision is plain code over the quoted amount; the model only
decides *when* to call ``place_order``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


class Decision(str, Enum):
    REFUSE = "refuse"  # over the spending limit
    CONFIRM = "confirm"  # within the limit, but at or above the confirmation threshold
    COMMIT = "commit"  # below the threshold: buy now


def decide(amount: Any, scope: Mapping[str, Any]) -> tuple[Decision, str]:
    """The decision for an order of ``amount`` under the user's ``scope``, and the reason."""
    if not isinstance(amount, (int, float)) or amount <= 0:
        return Decision.REFUSE, "the quote has no valid amount"
    limit, threshold = float(scope["max_purchase"]), float(scope["auto_confirm_under"])
    if amount > limit:
        return Decision.REFUSE, f"${amount:.2f} exceeds the user's ${limit:.2f} spending limit"
    if amount >= threshold:
        return Decision.CONFIRM, f"${amount:.2f} is at or above the ${threshold:.2f} confirmation threshold"
    return Decision.COMMIT, f"${amount:.2f} is below the ${threshold:.2f} confirmation threshold"
