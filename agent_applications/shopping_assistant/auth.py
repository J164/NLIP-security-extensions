"""OAuth 2.0 / OpenID Connect login and the NLIP session tickets it produces.

After an identity provider (Google, or the local mock) authenticates the user, the
orchestrator issues its own opaque ticket. The user client carries that ticket in every
NLIP request as an ECMA-430 ``authorization`` token submessage. Identity-provider tokens
never leave this module, and the ticket is never forwarded to subagents.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from secrets import token_urlsafe
from time import time
from typing import Any
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


@dataclass(frozen=True, slots=True)
class Identity:
    provider: str
    subject: str
    email: str
    name: str


@dataclass(frozen=True, slots=True)
class Ticket:
    value: str
    identity: Identity
    user_scope: dict[str, float]
    expires_at: float

    def user(self) -> dict[str, Any]:
        return {
            "provider": self.identity.provider,
            "subject": self.identity.subject,
            "email": self.identity.email,
            "name": self.identity.name,
            "user_scope": dict(self.user_scope),
        }


def scope_for(config: Mapping[str, Any], email: str) -> dict[str, float]:
    """The spending scope granted to this user by application policy."""
    policy = config.get("policy", {})
    scope = policy.get("users", {}).get(email.casefold()) or policy["default_scope"]
    return {
        "max_purchase": float(scope["max_purchase"]),
        "auto_confirm_under": float(scope["auto_confirm_under"]),
    }


class TicketStore:
    """In-memory map from opaque ticket to the authenticated user and their scope."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config
        self._ttl = float(config["auth"].get("ticket_ttl_seconds", 28800))
        self._tickets: dict[str, Ticket] = {}

    def issue(self, identity: Identity) -> Ticket:
        ticket = Ticket(
            token_urlsafe(32),
            identity,
            scope_for(self._config, identity.email),
            time() + self._ttl,
        )
        self._tickets[ticket.value] = ticket
        return ticket

    def resolve(self, value: str | None) -> Ticket | None:
        ticket = self._tickets.get(value or "")
        if ticket is None:
            return None
        if ticket.expires_at <= time():
            del self._tickets[ticket.value]
            return None
        return ticket

    def revoke(self, value: str | None) -> None:
        self._tickets.pop(value or "", None)


class IdentityProvider(ABC):
    name: str

    @abstractmethod
    async def login(self, request: Request) -> Response:
        """Redirect the browser to the provider's authorization endpoint."""

    @abstractmethod
    async def callback(self, request: Request) -> Identity:
        """Finish the authorization-code exchange and return the verified identity."""


class GoogleProvider(IdentityProvider):
    """Google OpenID Connect, authorization-code flow with PKCE (via Authlib)."""

    name = "google"

    def __init__(self, redirect_uri: str) -> None:
        from authlib.integrations.starlette_client import OAuth

        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET for Google login.")
        self.redirect_uri = redirect_uri
        oauth = OAuth()
        self.client = oauth.register(
            "google",
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=GOOGLE_DISCOVERY_URL,
            client_kwargs={"scope": "openid email profile", "code_challenge_method": "S256"},
        )

    async def login(self, request: Request) -> Response:
        return await self.client.authorize_redirect(request, self.redirect_uri)

    async def callback(self, request: Request) -> Identity:
        # Authlib checks state, exchanges the code, and validates the ID token signature,
        # issuer, audience, expiry and nonce.
        token = await self.client.authorize_access_token(request)
        claims = token["userinfo"]
        if not claims.get("email_verified"):
            raise PermissionError("Google account email is not verified.")
        return Identity("google", claims["sub"], claims["email"].casefold(), claims.get("name", ""))


class MockProvider(IdentityProvider):
    """A local stand-in for an OIDC provider, for graders and scripted evaluation.

    It mimics the redirect/code/state round trip but performs no real authentication.
    """

    name = "mock"

    def __init__(self, users: Mapping[str, Mapping[str, Any]], redirect_uri: str) -> None:
        self.users = {email.casefold(): dict(info) for email, info in users.items()}
        self.redirect_uri = redirect_uri
        self._codes: dict[str, str] = {}

    def identity(self, email: str) -> Identity:
        email = email.casefold()
        if email not in self.users:
            raise PermissionError(f"Unknown mock user {email!r}.")
        return Identity("mock", f"mock|{email}", email, self.users[email].get("name", email))

    async def login(self, request: Request) -> Response:
        state = token_urlsafe(16)
        request.session["mock_state"] = state
        return RedirectResponse(f"/auth/mock/authorize?{urlencode({'state': state})}")

    def authorize_page(self, state: str) -> HTMLResponse:
        buttons = "".join(
            f'<p><a class="user" href="{escape(self._issue(email, state))}">'
            f"Continue as {escape(info.get('name', email))} &lt;{escape(email)}&gt;</a></p>"
            for email, info in self.users.items()
        )
        return HTMLResponse(
            "<!doctype html><title>Mock sign-in</title>"
            "<body style='font-family:sans-serif;max-width:32rem;margin:3rem auto'>"
            "<h1>Mock identity provider</h1><p>Local test login. No real authentication.</p>"
            f"{buttons}</body>"
        )

    def _issue(self, email: str, state: str) -> str:
        code = token_urlsafe(16)
        self._codes[code] = email
        return f"{self.redirect_uri}?{urlencode({'code': code, 'state': state})}"

    async def callback(self, request: Request) -> Identity:
        state = request.session.pop("mock_state", None)
        if not state or request.query_params.get("state") != state:
            raise PermissionError("OAuth state mismatch.")
        email = self._codes.pop(request.query_params.get("code", ""), None)
        if email is None:
            raise PermissionError("Invalid or reused authorization code.")
        return self.identity(email)


def build_provider(config: Mapping[str, Any]) -> IdentityProvider:
    auth = config["auth"]
    provider = os.getenv("AUTH_PROVIDER", auth.get("provider", "mock")).casefold()
    redirect_uri = os.getenv("AUTH_REDIRECT_URI", auth["redirect_uri"])
    if provider == "google":
        return GoogleProvider(redirect_uri)
    if provider == "mock":
        return MockProvider(config.get("mock_users", {}), redirect_uri)
    raise ValueError(f"Unknown auth provider {provider!r}.")
