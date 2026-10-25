"""AgentSpec tool guard: ``before_action`` rules around the orchestrator's tool calls.

``stop`` refuses the call; ``user_inspection`` makes the user press Confirm first.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent_applications.shopping_assistant.tool_guard import GuardDecision, ToolCall, ToolGuard

from . import decision_log
from .engine import RuleEngine
from .predicates import PREDICATES, ActionContext

DEFAULT_RULES = Path(__file__).with_name("rules.spec")


class AgentSpecToolGuard(ToolGuard):
    def __init__(self, rules: str | None = None, entity: str = "orchestrator", **_: Any) -> None:
        self.engine = RuleEngine.from_file(rules or DEFAULT_RULES, PREDICATES)
        self.entity = entity

    async def before_tool_call(self, call: ToolCall) -> GuardDecision:
        started = time.perf_counter()
        context = ActionContext(call.name, call.args, call.user.get("user_scope") or {}, call.state)
        fired = self.engine.fired("before_action", context, action=call.name)
        ms = (time.perf_counter() - started) * 1000
        for rule in fired:
            if "stop" in rule.enforce:
                reason = f"@{rule.name}: {call.name} {dict(call.args)} is outside the user's scope"
                decision_log.write(self.entity, "agentspec_guard", "before_action", "block", rule.name, reason, ms)
                return GuardDecision.block(reason)
        for rule in fired:
            if "user_inspection" in rule.enforce:
                reason = f"@{rule.name}: {call.name} needs the user's confirmation"
                decision_log.write(self.entity, "agentspec_guard", "before_action",
                                   "require_confirmation", rule.name, reason, ms)
                return GuardDecision.require_confirmation(reason)
        decision_log.write(self.entity, "agentspec_guard", "before_action", "allow", None, None, ms)
        return GuardDecision.allow()
