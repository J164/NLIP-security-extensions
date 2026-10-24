"""Read-only product data used by the agents' tools."""

from __future__ import annotations

import json
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

from ._config import DATA_DIR


@cache
def products() -> dict[str, dict[str, Any]]:
    with (DATA_DIR / "catalog.json").open() as file:
        return {item["id"]: item for item in json.load(file)}


def _overlay_dir() -> Path | None:
    """Per-case content overrides used by the evaluation harness (unset in normal use)."""
    value = os.getenv("SHOP_OVERLAY_DIR")
    return Path(value) if value else None


@cache
def _base_reviews() -> dict[str, list[dict[str, Any]]]:
    with (DATA_DIR / "reviews.json").open() as file:
        return json.load(file)


def reviews() -> dict[str, list[dict[str, Any]]]:
    overlay = _overlay_dir()
    path = overlay / "reviews.json" if overlay else None
    if path is None or not path.is_file():
        return _base_reviews()
    with path.open() as file:
        return {**_base_reviews(), **json.load(file)}


def search(query: str, max_price: float | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Rank products by how many query words appear in their text."""
    words = [word for word in re.findall(r"[a-z0-9]+", query.casefold()) if len(word) > 1]
    scored = []
    for item in products().values():
        if max_price is not None and item["price"] > max_price:
            continue
        haystack = f"{item['title']} {item['category']} {item['description']}".casefold()
        score = sum(word in haystack for word in words)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["price"]))
    return [item for _, item in scored[:limit]]


def product_page(product_id: str) -> str | None:
    """The fetched product web page. This text is third-party content, not trusted."""
    overlay = _overlay_dir()
    if overlay and (overlay / "pages" / f"{product_id}.txt").is_file():
        return (overlay / "pages" / f"{product_id}.txt").read_text()
    path = DATA_DIR / "pages" / f"{product_id}.txt"
    if path.is_file():
        return path.read_text()
    item = products().get(product_id)
    if item is None:
        return None
    return f"{item['title']} — Product page\n\n{item['description']}\nPrice: ${item['price']:.2f}"
