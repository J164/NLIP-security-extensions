"""Web chat UI and OAuth login routes, mounted on the orchestrator's FastAPI app.

The browser page is itself the NLIP user client: its JavaScript posts NLIP JSON to
``/nlip`` with the session ticket in an ``authorization`` token submessage.
"""

from __future__ import annotations

import os
from html import escape
from pathlib import Path
from secrets import token_urlsafe

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from .auth import IdentityProvider, MockProvider, TicketStore

STATIC_DIR = Path(__file__).with_name("static")


class MockTokenRequest(BaseModel):
    email: str


def install(app: FastAPI, tickets: TicketStore, provider: IdentityProvider) -> None:
    # The cookie only holds the ticket id and OAuth state, signed with SESSION_SECRET.
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("SESSION_SECRET") or token_urlsafe(32),
        same_site="lax",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/login")
    async def login(request: Request):
        return await provider.login(request)

    @app.get("/auth/callback")
    async def callback(request: Request):
        try:
            identity = await provider.callback(request)
        except Exception as error:  # Any failed exchange is a failed login.
            return HTMLResponse(
                f"<p>Login failed: {escape(str(error))}</p><p><a href='/'>Back</a></p>",
                status_code=400,
            )
        tickets.revoke(request.session.get("ticket"))
        request.session["ticket"] = tickets.issue(identity).value
        return RedirectResponse("/")

    @app.get("/auth/session")
    async def auth_session(request: Request) -> dict:
        ticket = tickets.resolve(request.session.get("ticket"))
        if ticket is None:
            raise HTTPException(401, "Not signed in.")
        return {"ticket": ticket.value, "user": ticket.user(), "provider": provider.name}

    @app.get("/logout")
    async def logout(request: Request) -> RedirectResponse:
        tickets.revoke(request.session.get("ticket"))
        request.session.clear()
        return RedirectResponse("/")

    if isinstance(provider, MockProvider):

        @app.get("/auth/mock/authorize", include_in_schema=False)
        async def mock_authorize(state: str) -> HTMLResponse:
            return provider.authorize_page(state)

        @app.post("/auth/mock/token")
        async def mock_token(body: MockTokenRequest) -> dict:
            """Scripted-client login for tests and evaluation (mock provider only)."""
            try:
                ticket = tickets.issue(provider.identity(body.email))
            except PermissionError as error:
                raise HTTPException(403, str(error)) from error
            return {"ticket": ticket.value, "user": ticket.user()}
