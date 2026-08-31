"""Configurable security checks for messages entering or leaving an NLIP entity."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Literal

from ..nlip import NLIPMessage
from ..session import NLIPSession


class EnforcementPoint(str, Enum):
    """The two directions exposed to security extension authors."""

    INGRESS = "ingress"
    EGRESS = "egress"


# NOTE: Developers may add actions here, but must also implement them in enforce().
class SecurityAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class SecurityDecision:
    """An extension's explicit decision about the current message."""

    action: SecurityAction
    message: NLIPMessage | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, SecurityAction):
            raise ValueError("SecurityDecision action must be a SecurityAction.")
        if self.message is not None and not isinstance(self.message, NLIPMessage):
            raise ValueError("SecurityDecision message must be an NLIPMessage.")
        if self.action is SecurityAction.REPLACE and self.message is None:
            raise ValueError("A REPLACE decision requires an NLIP message.")
        if self.action is not SecurityAction.REPLACE and self.message is not None:
            raise ValueError("Only a REPLACE decision may contain an NLIP message.")

    @classmethod
    def allow(cls) -> "SecurityDecision":
        return cls(SecurityAction.ALLOW)

    @classmethod
    def block(cls, reason: str) -> "SecurityDecision":
        return cls(SecurityAction.BLOCK, reason=reason)

    @classmethod
    def replace(
        cls, message: NLIPMessage, reason: str | None = None
    ) -> "SecurityDecision":
        return cls(SecurityAction.REPLACE, message=message, reason=reason)


@dataclass(frozen=True, slots=True)
class SecurityEvent:
    """What an extension can observe at one ingress or egress point.

    ``session.history`` contains completed earlier turns. ``related_message`` is the
    request paired with a response event and is otherwise absent.
    """

    point: EnforcementPoint
    message: NLIPMessage
    session: NLIPSession
    local_entity: str
    peer: str | None
    component: Literal["client", "server"]
    related_message: NLIPMessage | None = None


class SecurityExtension(ABC):
    """Base class implemented by every NLIP security extension."""

    @abstractmethod
    async def evaluate(self, event: SecurityEvent) -> SecurityDecision:
        """Inspect one event and explicitly allow, block, or replace its message."""


class SecurityExtensionConfigError(ValueError):
    """Raised when configured extensions cannot be loaded safely at startup."""


class SecurityExtensionExecutionError(RuntimeError):
    """Raised when an extension fails or returns an invalid decision."""


class SecurityBlockedError(PermissionError):
    """Raised when an extension blocks an NLIP message."""

    def __init__(self, extension: str, point: EnforcementPoint, reason: str) -> None:
        self.extension = extension
        self.point = point
        self.reason = reason
        super().__init__(f"{extension} blocked {point.value}: {reason}")


class SecurityExtensionManager:
    """Load enabled extensions and invoke them in configured order."""

    def __init__(self, local_entity: str) -> None:
        local_entity = local_entity.strip().casefold()
        if not local_entity:
            raise ValueError("Security extension manager entity cannot be empty.")
        self.local_entity = local_entity
        self._extensions: dict[
            EnforcementPoint, list[tuple[str, SecurityExtension]]
        ] = {point: [] for point in EnforcementPoint}

    @property
    def enabled(self) -> bool:
        return any(self._extensions.values())

    def extension_names(self, point: EnforcementPoint) -> tuple[str, ...]:
        return tuple(name for name, _ in self._extensions[point])

    def register(
        self,
        name: str,
        extension: SecurityExtension,
        points: Sequence[EnforcementPoint],
    ) -> None:
        """Register one instance at one or both ordered enforcement points."""
        if not isinstance(extension, SecurityExtension):
            raise TypeError("extension must be a SecurityExtension instance.")
        if not name or not points:
            raise ValueError("An extension needs a name and at least one point.")
        if not all(isinstance(point, EnforcementPoint) for point in points):
            raise ValueError("Unknown security enforcement point.")
        for point in points:
            if any(existing == name for existing, _ in self._extensions[point]):
                raise ValueError(f"Duplicate {point.value} extension: {name}.")
            self._extensions[point].append((name, extension))

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, entity: str
    ) -> "SecurityExtensionManager":
        """Build an entity's ordered pipelines from parsed TOML configuration."""
        manager = cls(entity)
        entities = config.get("entities")
        if not isinstance(entities, Mapping) or entity not in entities:
            raise SecurityExtensionConfigError(f"Unknown NLIP entity: {entity!r}.")
        entity_config = entities[entity]
        if not isinstance(entity_config, Mapping):
            raise SecurityExtensionConfigError(f"entities.{entity} must be a table.")

        security = entity_config.get("security")
        if security is None:
            return manager
        if not isinstance(security, Mapping):
            raise SecurityExtensionConfigError(
                f"entities.{entity}.security must be a table."
            )
        unknown_points = set(security) - {point.value for point in EnforcementPoint}
        if unknown_points:
            names = ", ".join(sorted(str(point) for point in unknown_points))
            raise SecurityExtensionConfigError(
                f"Unknown security point(s) for {entity}: {names}."
            )

        definitions = config.get("security_extensions", {})
        if not isinstance(definitions, Mapping):
            raise SecurityExtensionConfigError("security_extensions must be a table.")

        instances: dict[str, SecurityExtension] = {}
        for point in EnforcementPoint:
            names = security.get(point.value, [])
            if not isinstance(names, list) or not all(
                isinstance(name, str) and name for name in names
            ):
                raise SecurityExtensionConfigError(
                    f"entities.{entity}.security.{point.value} must be a list of names."
                )
            for name in names:
                if name not in instances:
                    instances[name] = cls._load(name, definitions)
                try:
                    manager.register(name, instances[name], [point])
                except ValueError as error:
                    raise SecurityExtensionConfigError(str(error)) from error
        return manager

    @staticmethod
    def _load(
        name: str, definitions: Mapping[str, Any]
    ) -> SecurityExtension:
        definition = definitions.get(name)
        if not isinstance(definition, Mapping):
            raise SecurityExtensionConfigError(
                f"Security extension {name!r} has no definition."
            )
        entrypoint = definition.get("entrypoint")
        if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
            raise SecurityExtensionConfigError(
                f"security_extensions.{name}.entrypoint must be 'module:Class'."
            )
        module_name, class_name = entrypoint.split(":")
        options = definition.get("options", {})
        if not isinstance(options, Mapping):
            raise SecurityExtensionConfigError(
                f"security_extensions.{name}.options must be a table."
            )
        try:
            extension_class = getattr(import_module(module_name), class_name)
            if not isinstance(extension_class, type) or not issubclass(
                extension_class, SecurityExtension
            ):
                raise TypeError("entrypoint is not a SecurityExtension class")
            return extension_class(**dict(options))
        except Exception as error:
            raise SecurityExtensionConfigError(
                f"Could not load security extension {name!r} from {entrypoint!r}: "
                f"{error}"
            ) from error

    async def enforce(
        self,
        point: EnforcementPoint,
        message: NLIPMessage,
        session: NLIPSession,
        *,
        component: Literal["client", "server"],
        related_message: NLIPMessage | None = None,
    ) -> NLIPMessage:
        """Run one ordered point, failing closed on extension errors."""
        current = message.model_copy(deep=True)
        for name, extension in self._extensions[point]:
            event = SecurityEvent(
                point=point,
                message=current.model_copy(deep=True),
                session=session,
                local_entity=self.local_entity,
                peer=session.peer,
                component=component,
                related_message=(
                    related_message.model_copy(deep=True)
                    if related_message is not None
                    else None
                ),
            )
            try:
                decision = await extension.evaluate(event)
            except Exception as error:
                raise SecurityExtensionExecutionError(
                    f"Security extension {name!r} failed at {point.value}: {error}"
                ) from error
            if not isinstance(decision, SecurityDecision):
                raise SecurityExtensionExecutionError(
                    f"Security extension {name!r} returned no valid decision."
                )
            if decision.action is SecurityAction.BLOCK:
                raise SecurityBlockedError(
                    name, point, decision.reason or "no reason provided"
                )
            if decision.action is SecurityAction.REPLACE:
                assert decision.message is not None
                current = decision.message.model_copy(deep=True)
        return current

    async def enforce_ingress(
        self,
        message: NLIPMessage,
        session: NLIPSession,
        *,
        component: Literal["client", "server"],
        related_message: NLIPMessage | None = None,
    ) -> NLIPMessage:
        return await self.enforce(
            EnforcementPoint.INGRESS,
            message,
            session,
            component=component,
            related_message=related_message,
        )

    async def enforce_egress(
        self,
        message: NLIPMessage,
        session: NLIPSession,
        *,
        component: Literal["client", "server"],
        related_message: NLIPMessage | None = None,
    ) -> NLIPMessage:
        return await self.enforce(
            EnforcementPoint.EGRESS,
            message,
            session,
            component=component,
            related_message=related_message,
        )
