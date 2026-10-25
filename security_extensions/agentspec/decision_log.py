"""Optional JSON-lines log of every rule decision, at $SHOP_SECURITY_LOG when it is set.

Row: {"ts", "entity", "extension", "point", "action", "rule", "reason", "ms"}, the format
the benchmark harness reads for replace decisions and SE overhead.
"""

from __future__ import annotations

import json
import os
import time


def write(entity: str, extension: str, point: str, action: str, rule: str | None,
          reason: str | None, ms: float) -> None:
    path = os.getenv("SHOP_SECURITY_LOG")
    if not path:
        return
    row = {"ts": time.time(), "entity": entity, "extension": extension, "point": point,
           "action": action, "rule": rule, "reason": reason, "ms": round(ms, 4)}
    with open(path, "a") as file:
        file.write(json.dumps(row) + "\n")
