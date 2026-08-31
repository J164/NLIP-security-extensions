"""HTTP client for the ECMA-431 NLIP binding."""

from collections.abc import Collection, Mapping
from secrets import token_urlsafe
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..nlip import AllowedFormat, NLIPMessage, NLIPSubMessage
from ..nlip_security_extensions import SecurityExtensionManager
from ..session import NLIPSession, NLIPSessionTurn


def _nlip_endpoint(url: str) -> str:
    """Validate an HTTP base URL and normalize it to the ``/nlip`` endpoint."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("NLIP endpoint must be an http:// or https:// URL.")
    if parsed.query or parsed.fragment:
        raise ValueError("NLIP endpoint must not contain a query or fragment.")

    path = parsed.path.rstrip("/")
    if not path.casefold().endswith("/nlip"):
        path = f"{path}/nlip"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class NLIPClientSession:
    """A conversation with one destination, including its own tokens and history."""

    def __init__(self, client: "NLIPClient", destination: str) -> None:
        self._client = client
        self.destination = client._destination(destination)
        self.context = NLIPSession(
            token_urlsafe(24),
            peer=self.destination,
            max_history=client.max_history,
        )
        # The peer must echo this client-created token and may add its own token.
        self._tokens = [
            NLIPSubMessage(
                format=AllowedFormat.token,
                subformat=client._conversation_token_type,
                content=self.context.id, # Here the conversation token is the NLIPSession id.
            )
        ]

    @property
    def id(self) -> str:
        return self.context.id

    @property
    def history(self) -> tuple[NLIPSessionTurn, ...]:
        return self.context.history

    def _prepare(self, message: NLIPMessage) -> NLIPMessage:
        # Never mutate the caller's message while adding this session's tokens.
        outgoing = message.model_copy(deep=True)
        for token in self._tokens:
            if not outgoing.has_token(token):
                outgoing.add_submessage(token.model_copy(deep=True))
        return outgoing

    @staticmethod
    def _parse_response(response: httpx.Response) -> NLIPMessage:
        response.raise_for_status()
        return NLIPMessage.model_validate(response.json())

    def _accept_response(
        self, outgoing: NLIPMessage, message: NLIPMessage
    ) -> NLIPMessage:
        for token in message.token_submessages():
            if not any(
                saved.subformat.casefold() == token.subformat.casefold()
                and saved.content == token.content
                and saved.label == token.label
                for saved in self._tokens
            ):
                self._tokens.append(token.model_copy(deep=True))
        self.context.record(outgoing, message)
        return message

    async def send(self, message: NLIPMessage) -> NLIPMessage:
        """Run one message through this conversation's complete client pipeline."""
        outgoing = self._prepare(message)
        outgoing = await self._client.security.enforce_egress(
            outgoing, self.context, component="client"
        )
        # A replacement may change content, but transport-owned session tokens remain.
        outgoing = self._prepare(outgoing)
        response = await self._client._post(self.destination, outgoing) # Send the message to the destination endpoint.
        incoming = self._parse_response(response)
        incoming = await self._client.security.enforce_ingress(
            incoming,
            self.context,
            component="client",
            related_message=outgoing,
        )
        return self._accept_response(outgoing, incoming)

    async def send_text(
        self, content: str, language: str = "English"
    ) -> NLIPMessage:
        return await self.send(NLIPMessage.text(content, language))


class NLIPClient:
    """Send NLIP messages to any named destination known at construction time.

    The client is a small directory plus transport helper. It does not represent a
    connection to one peer; callers select the peer on every ``send`` operation.
    """

    TOOL_NAME = "send_nlip_message"

    def __init__(
        self,
        destinations: Mapping[str, str],
        timeout: float = 120.0,
        *,
        identity: str = "client",
        max_history: int = 100,
        security: SecurityExtensionManager | None = None,
    ) -> None:
        if not destinations:
            raise ValueError("NLIPClient requires at least one destination.")

        # Names are normalized once so config names and model-selected names match
        # case-insensitively during every later send.
        self.endpoints: dict[str, str] = {}
        for name, endpoint in destinations.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("NLIP destination names must be non-empty strings.")
            if not isinstance(endpoint, str):
                raise ValueError(f"NLIP endpoint for {name!r} must be a URL string.")
            normalized_name = name.strip().casefold()
            if normalized_name in self.endpoints:
                raise ValueError(f"Duplicate NLIP destination name: {name!r}.")
            self.endpoints[normalized_name] = _nlip_endpoint(endpoint)

        identity = identity.strip().casefold()
        if not identity:
            raise ValueError("NLIP client identity cannot be empty.")
        if max_history < 1:
            raise ValueError("NLIP client max_history must be positive.")
        if security is not None and security.local_entity != identity:
            raise ValueError("NLIP client and security manager identities must match.")
        self.identity = identity
        self.security = security or SecurityExtensionManager(identity)
        self.timeout = timeout
        self.max_history = max_history
        self._conversation_token_type = f"conversation_{identity}"
        # Key: (scope, destination).
        #   scope is a client-local owner id (normally an inbound NLIPSession.id);
        #   None means the client's default conversation for that destination.
        #   destination is the normalized entity name from ``self.endpoints``.
        # Value: the reusable point-to-point conversation with that destination,
        # including its protocol tokens and bounded request/response history.
        self._sessions: dict[tuple[str | None, str], NLIPClientSession] = {}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        exclude: Collection[str] = (),
        timeout: float = 120.0,
        identity: str = "client",
        max_history: int = 100,
        security: SecurityExtensionManager | None = None,
    ) -> "NLIPClient":
        """Build from ``[entities.<name>].endpoint``, optionally omitting the caller."""
        entities = config.get("entities")
        if not isinstance(entities, Mapping) or not entities:
            raise ValueError("NLIP config must contain a non-empty 'entities' table.")

        excluded = {name.casefold() for name in exclude}
        destinations: dict[str, str] = {}
        for name, entity in entities.items():
            if not isinstance(name, str) or not isinstance(entity, Mapping):
                raise ValueError("Each NLIP entity must be a named table.")
            endpoint = entity.get("endpoint")
            if not isinstance(endpoint, str):
                raise ValueError(f"entities.{name}.endpoint must be a URL string.")
            if name.casefold() not in excluded:
                destinations[name] = endpoint
        return cls(
            destinations,
            timeout=timeout,
            identity=identity,
            max_history=max_history,
            security=security,
        )

    @property
    def destinations(self) -> tuple[str, ...]:
        """Names accepted by all send methods and exposed to agent tools."""
        return tuple(self.endpoints)

    @property
    def tool_description(self) -> str:
        """Framework-neutral description generated from the current directory."""
        names = ", ".join(self.destinations)
        return (
            "Send a text message to another entity over NLIP and return its response. "
            f"Available destinations: {names}."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        """JSON Schema shared by framework-specific tool adapters."""
        return {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": list(self.destinations),
                    "description": "Name of the destination entity.",
                },
                "message": {
                    "type": "string",
                    "description": "Text to send to the destination entity.",
                },
            },
            "required": ["destination", "message"],
            "additionalProperties": False,
        }

    def _destination(self, destination: str) -> str:
        if not isinstance(destination, str):
            raise ValueError("NLIP destination must be a string.")
        name = destination.strip().casefold()
        if name not in self.endpoints:
            available = ", ".join(self.destinations)
            raise ValueError(
                f"Unknown NLIP destination {destination!r}. Available: {available}."
            )
        return name

    def create_session(self, destination: str) -> NLIPClientSession:
        """Create an independent conversation, even with an already-used destination."""
        return NLIPClientSession(self, destination)

    def session(
        self, destination: str, *, scope: str | None = None
    ) -> NLIPClientSession:
        """Reuse one destination session within an optional parent-session scope."""
        destination = self._destination(destination)
        if scope is not None and (not isinstance(scope, str) or not scope):
            raise ValueError("NLIP client session scope must be a non-empty string.")
        key = (scope, destination)
        if key not in self._sessions:
            self._sessions[key] = self.create_session(destination)
        return self._sessions[key]

    def clear_scope(self, scope: str) -> None:
        """Forget outbound conversations owned by a finished parent session."""
        if not isinstance(scope, str) or not scope:
            raise ValueError("NLIP client session scope must be a non-empty string.")
        for key in [key for key in self._sessions if key[0] == scope]:
            del self._sessions[key]

    async def _post(self, destination: str, message: NLIPMessage) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(
                self.endpoints[destination], json=message.to_dict()
            )

    async def send(
        self,
        destination: str,
        message: NLIPMessage,
        *,
        scope: str | None = None,
    ) -> NLIPMessage:
        return await self.session(destination, scope=scope).send(message)

    async def send_text(
        self,
        destination: str,
        content: str,
        language: str = "English",
        *,
        scope: str | None = None,
    ) -> NLIPMessage:
        return await self.session(destination, scope=scope).send(
            NLIPMessage.text(content, language)
        )

    async def send_nlip_message(
        self, destination: str, message: str, *, scope: str | None = None
    ) -> str:
        """Agent-facing helper that turns a text exchange into a plain tool result."""
        response = await self.session(destination, scope=scope).send(
            NLIPMessage.text(message)
        )
        return response.extract_text(language=None) or response.to_json()
