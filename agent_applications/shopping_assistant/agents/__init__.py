"""NLIP servers for the shopping assistant's subagents."""

from __future__ import annotations

from fastapi import FastAPI

from .._config import CONFIG


def serve(app: FastAPI, entity: str) -> None:
    import uvicorn

    settings = CONFIG["entities"][entity]
    uvicorn.run(app, host=settings["host"], port=settings["port"])
