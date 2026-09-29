"""Tool-boundary hook: checks invoked directly around the orchestrator's tool calls.

``SecurityExtensionManager`` only sees NLIP ingress and egress. Security functions that
must decide on the tool call itself (a Type-3 SE invoked "immediately before or after a
tool or model call") plug in here. Guards are configured like security extensions::

    [tool_guards.my_guard]
    entrypoint = "security_extensions.my_se:MyToolGuard"

    [entities.orchestrator]
    tool_guards = ["my_guard"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import Any


class GuardAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass(frozen=True, slots=True)
class GuardDecision:
    action: GuardAction
    reason: str | None = None

    @classmethod
    def allow(cls) -> "GuardDecision":
        return cls(GuardAction.ALLOW)

    @classmethod
    def block(cls, reason: str) -> "GuardDecision":
        return cls(GuardAction.BLOCK, reason)

    @classmethod
    def require_confirmation(cls, reason: str) -> "GuardDecision":
        return cls(GuardAction.REQUIRE_CONFIRMATION, reason)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Everything a guard may see: the call, the authenticated user, and order state."""

    name: str
    args: dict[str, Any]
    session_id: str
    user: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)


class ToolGuard(ABC):
    @abstractmethod
    async def before_tool_call(self, call: ToolCall) -> GuardDecision:
        """Allow, block, or require user confirmation before the tool runs."""

    async def after_tool_call(self, call: ToolCall, result: str) -> str:
        """Inspect or rewrite the tool result before the model sees it."""
        return result


def load_tool_guards(config: Mapping[str, Any], entity: str) -> list[tuple[str, ToolGuard]]:
    """Instantiate the entity's enabled guards in configured order."""
    names = config["entities"][entity].get("tool_guards", [])
    definitions = config.get("tool_guards", {})
    guards = []
    for name in names:
        definition = definitions.get(name)
        if not isinstance(definition, Mapping) or ":" not in str(definition.get("entrypoint")):
            raise ValueError(f"tool_guards.{name}.entrypoint must be 'module:Class'.")
        module_name, class_name = definition["entrypoint"].split(":")
        guard_class = getattr(import_module(module_name), class_name)
        guard = guard_class(**dict(definition.get("options", {})))
        if not isinstance(guard, ToolGuard):
            raise TypeError(f"{definition['entrypoint']} is not a ToolGuard.")
        guards.append((name, guard))
    return guards
