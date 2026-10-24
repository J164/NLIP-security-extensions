"""Pass-through NLIP proxy placed at a subagent's configured endpoint.

    python -m adversarial_tests.agentspec_benchmark.proxy --port 8001 \
        --upstream http://127.0.0.1:8011 --target search_agent

The orchestrator keeps dialling the address in its config; the real subagent listens on an
internal port behind this proxy. Each request is forwarded unchanged and each reply is
returned unchanged, except for the operations the harness armed for the current case
(``cases/schema.md``, "Proxy"). The proxy contains no attack logic of its own.

This is a raw HTTP forwarder rather than an ``NLIPServer``: an NLIP server would add its own
conversation token to every reply, so even an unarmed proxy would alter the traffic.

Control routes, accepted from 127.0.0.1 only:

- ``POST /_bench/arm``  body ``{"case_id": str, "ops": [...]}``: set the operations for the
  next case (``[]`` for pure pass-through) and clear the log.
- ``GET /_bench/log``: ``{"case_id", "received": [<request dict>...], "applied": [...]}``,
  what the orchestrator sent and which operations were applied to which reply.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from nlip_sdk import AllowedFormat, NLIPMessage, NLIPSubMessage

UPSTREAM_TIMEOUT_S = 300.0
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


# --- operations ---------------------------------------------------------------------------

def _content_slots(message: NLIPMessage) -> list[int]:
    """Indexes of the non-token submessages.

    Positions count only content parts, so ``"last"`` lands before the session tokens the
    server appends, where a subagent's own handler would have put it.
    """
    return [n for n, part in enumerate(message.submessages or []) if part.format != AllowedFormat.token]


def _insert_index(message: NLIPMessage, position: Any) -> int:
    slots = _content_slots(message)
    count = len(message.submessages or [])
    if position == "first":
        return slots[0] if slots else 0
    if position == "last" or not slots or position >= len(slots):
        return slots[-1] + 1 if slots else count
    return slots[position]


def _set_path(content: Any, path: Sequence[str | int], value: Any) -> Any:
    """A copy of ``content`` with the nested value at ``path`` set to ``value``."""
    content = copy.deepcopy(content)
    target = content
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return content


def apply_ops(message: NLIPMessage, ops: Sequence[Mapping[str, Any]]) -> tuple[NLIPMessage, list[dict]]:
    """Apply case operations to a copy of a reply. Returns the new reply and per-op outcomes."""
    message = message.model_copy(deep=True)
    outcomes = []
    for op in ops:
        kind = op["op"]
        outcome = {"op": kind, "applied": True}
        if kind == "add_submessage":
            part = NLIPSubMessage(
                format=op["format"], subformat=op["subformat"], content=op["content"],
                label=op.get("label"),
            )
            parts = list(message.submessages or [])
            parts.insert(_insert_index(message, op.get("position", "last")), part)
            message.submessages = parts
        elif kind == "set_primary_text":
            if message.format != AllowedFormat.text:
                message.format, message.subformat = AllowedFormat.text, "English"
            message.content = op["text"]
        elif kind == "replace_field":
            matches = [p for p in message.submessages or [] if p.label == op["label"]]
            outcome["matches"] = len(matches)
            if not matches:
                outcome.update(applied=False, reason=f"no submessage labeled {op['label']!r}")
            for part in matches:
                try:
                    part.content = (
                        _set_path(part.content, op["path"], op["value"]) if op.get("path") else op["value"]
                    )
                except (KeyError, IndexError, TypeError) as error:
                    outcome.update(applied=False, reason=f"path not found: {error!r}")
        else:
            outcome.update(applied=False, reason=f"unknown op {kind!r}")
        outcomes.append(outcome)
    return message, outcomes


# --- server -------------------------------------------------------------------------------

def create_app(upstream: str, target: str, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """``transport`` lets tests route upstream calls to an in-process app."""
    upstream = upstream.rstrip("/")
    app = FastAPI(title=f"benchmark proxy for {target}")
    state: dict[str, Any] = {"case_id": None, "ops": [], "received": [], "applied": []}

    def local_only(request: Request) -> None:
        if request.client is None or request.client.host not in LOCAL_HOSTS:
            raise HTTPException(403, "control routes are local only")

    @app.get("/health")
    async def health() -> Response:
        async with httpx.AsyncClient(timeout=5, transport=transport) as client:
            try:
                upstream_health = await client.get(f"{upstream}/health")
            except httpx.HTTPError as error:
                return JSONResponse({"status": "upstream down", "error": str(error)}, 503)
        return JSONResponse({"status": "ok", "target": target}, upstream_health.status_code)

    @app.post("/_bench/arm")
    async def arm(request: Request) -> dict:
        local_only(request)
        body = await request.json()
        state.update(case_id=body.get("case_id"), ops=list(body.get("ops") or []), received=[], applied=[])
        return {"armed": len(state["ops"]), "case_id": state["case_id"]}

    @app.get("/_bench/log")
    async def log(request: Request) -> dict:
        local_only(request)
        return {k: state[k] for k in ("case_id", "received", "applied")}

    @app.post("/nlip")
    async def nlip(request: Request) -> Response:
        body = await request.body()
        try:
            state["received"].append(await request.json())
        except ValueError:
            state["received"].append({"unparsed": body.decode(errors="replace")})
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S, transport=transport) as client:
            reply = await client.post(
                f"{upstream}/nlip", content=body, headers={"content-type": "application/json"}
            )
        if reply.status_code != 200 or not state["ops"]:
            return Response(reply.content, reply.status_code, media_type="application/json")
        message, outcomes = apply_ops(NLIPMessage.model_validate(reply.json()), state["ops"])
        state["applied"].append({"reply": len(state["received"]) - 1, "ops": outcomes})
        return JSONResponse(message.to_dict())

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True, help="port the orchestrator dials")
    parser.add_argument("--upstream", required=True, help="real subagent base URL")
    parser.add_argument("--target", required=True, help="subagent name, for logs")
    args = parser.parse_args()
    uvicorn.run(create_app(args.upstream, args.target), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
