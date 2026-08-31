"""Load the shared configuration used by NLIP applications.

Usage::

    config = load_config("config.toml")
    client = NLIPClient.from_config(config, identity="my_agent")
    security = SecurityExtensionManager.from_config(config, entity="my_agent")

The SDK-defined portion contains ``[entities.<name>]`` routing entries and optional
``[entities.<name>.security]`` / ``[security_extensions.<name>]`` tables. Applications
may add their own fields, such as server ``host``/``port`` or a user's default target.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load TOML and validate the entity routing table shared by SDK components."""
    with Path(path).open("rb") as config_file:
        config = tomllib.load(config_file)

    entities = config.get("entities")
    if not isinstance(entities, dict) or not entities:
        raise ValueError("NLIP config must define at least one [entities.<name>] table.")
    for name, entity in entities.items():
        if not isinstance(entity, dict):
            raise ValueError(f"entities.{name} must be a TOML table.")
        endpoint = entity.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(f"entities.{name}.endpoint must be a non-empty URL string.")
    return config
