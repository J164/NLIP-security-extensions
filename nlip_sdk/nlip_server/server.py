"""FastAPI server for the ECMA-431 NLIP binding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from secrets import token_urlsafe
from typing import TypeAlias

from fastapi import FastAPI, HTTPException

from ..nlip import NLIPMessage, ReservedToken
from ..nlip_security_extensions import (
    SecurityBlockedError,
    SecurityExtensionManager,
)
from ..session import NLIPSession

NLIPHandlerResult: TypeAlias = NLIPMessage | str


class NLIPServer(ABC):
    """Stable HTTP pipeline around an agent-specific ``handle`` implementation.

    Subclasses own the model/framework lifecycle and message handling. This class owns
    protocol behavior, security enforcement, and the FastAPI boundary once for every
    kind of agent.
    """

    def __init__(
        self,
        identity: str = "server",
        *,
        max_sessions: int = 128,
        max_history: int = 100,
        security: SecurityExtensionManager | None = None,
    ) -> None:
        identity = identity.strip().casefold()
        if not identity:
            raise ValueError("NLIP server identity cannot be empty.")
        if max_sessions < 1 or max_history < 1:
            raise ValueError("NLIP session and history limits must be positive.")
        if security is not None and security.local_entity != identity:
            raise ValueError("NLIP server and security manager identities must match.")
        self.identity = identity
        self.security = security or SecurityExtensionManager(identity)
        self._conversation_token_type = f"conversation_{identity}"
        self._max_sessions = max_sessions
        self._max_history = max_history
        # ponytail: process-local storage is enough for this demo; use a shared store
        # before running multiple workers or requiring sessions to survive restarts.
        # Key: the opaque server-issued conversation token (also NLIPSession.id)
        # that a client returns in later requests to resume the conversation.
        # Value: this server's context for that conversation, including its lock,
        # activity timestamp, and bounded request/response history.
        self._sessions: dict[str, NLIPSession] = {}

    async def startup(self) -> None:
        """Optionally initialize the wrapped agent when the FastAPI app starts."""

    async def shutdown(self) -> None:
        """Optionally release the wrapped agent when the FastAPI app stops."""

    async def session_started(self, session: NLIPSession) -> None:
        """Optionally allocate model/framework state for a new conversation."""

    async def session_ended(self, session: NLIPSession) -> None:
        """Optionally release model/framework state for an evicted conversation."""

    @abstractmethod
    async def handle(
        self, request: NLIPMessage, session: NLIPSession
    ) -> NLIPHandlerResult:
        """Produce a response body; the shared pipeline finalizes protocol metadata."""

    async def _end_session(self, session: NLIPSession) -> None:
        self._sessions.pop(session.id, None)
        await self.session_ended(session)

    async def _session_for(self, request: NLIPMessage) -> NLIPSession:
        """Resume the request's server-side session, or create a new one.

        A request resumes a session only when it carries this server's token type
        (``conversation_<server identity>``) and the opaque token value still exists
        as a key in ``self._sessions``. Tokens issued by other entities are ignored.
        A missing, unknown, expired, or post-restart token starts a new session.
        """
        # Search all token parts because one NLIP message may carry conversation
        # tokens issued by several entities. Only this server's token identifies
        # an entry in this server's process-local session store.
        for token in request.token_submessages():
            if token.subformat.casefold() == self._conversation_token_type:
                session = self._sessions.get(str(token.content))
                if session is not None:
                    session.touch()
                    return session

        # Make room before creating a session; eviction also releases any
        # framework-specific conversation state through session_ended().
        if len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda item: item.last_active)
            await self._end_session(oldest)

        # The new opaque id is both NLIPSession.id and the dictionary key. Later,
        # process_exchange() returns it to the client as this server's token.
        session = NLIPSession(token_urlsafe(24), max_history=self._max_history)
        self._sessions[session.id] = session
        try:
            # Subclasses may allocate one model conversation for this NLIP session.
            await self.session_started(session)
        except Exception:
            self._sessions.pop(session.id, None)
            raise
        return session

    async def process_exchange(self, request: NLIPMessage) -> NLIPMessage:
        """Process one complete request-response turn within an NLIP session.

        This is the protocol-level entry point: it resolves the session, runs both
        security checkpoints, invokes the agent-specific ``handle()``, restores
        required protocol metadata, and records the finalized turn.
        """
        session = await self._session_for(request)
        # Serialize the whole turn so agent state, security checks, and history cannot
        # observe different orders for concurrent requests in the same conversation.
        async with session.lock:
            wire_request = request
            request = await self.security.enforce_ingress(
                request, session, component="server"
            )
            # Subclasses implement only the agent response; this method owns the
            # surrounding NLIP session, security, and protocol behavior.
            result = await self.handle(request, session)
            response = NLIPMessage.text(result) if isinstance(result, str) else result

            # Keep protocol-required response behavior centralized in the server.
            response.echo_tokens_from(wire_request)
            # The client returns this opaque token to resume the server's state.
            response.add_token(session.id, self._conversation_token_type)
            if wire_request.is_control():
                response.messagetype = ReservedToken.control.value

            response = await self.security.enforce_egress(
                response, session, component="server", related_message=request
            )
            # A replacement may omit protocol metadata, so restore it afterwards.
            response.echo_tokens_from(wire_request)
            response.add_token(session.id, self._conversation_token_type)
            if wire_request.is_control():
                response.messagetype = ReservedToken.control.value

            session.record(request, response)
            return response

    @asynccontextmanager
    async def _lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        # Tie model resources (for example the Claude subprocess) to the HTTP app.
        await self.startup()
        try:
            yield
        finally:
            try:
                for session in list(self._sessions.values()):
                    await self._end_session(session)
            finally:
                await self.shutdown()

    def create_nlip_app(self) -> FastAPI:
        """Create a FastAPI app exposing the ECMA-431 ``POST /nlip`` endpoint."""
        app = FastAPI(lifespan=self._lifespan)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/nlip")
        async def post_nlip(request: NLIPMessage) -> dict:
            # FastAPI validates JSON into NLIPMessage before the shared pipeline runs.
            try:
                return (await self.process_exchange(request)).to_dict()
            except SecurityBlockedError as error:
                raise HTTPException(status_code=403, detail=str(error)) from error

        return app
