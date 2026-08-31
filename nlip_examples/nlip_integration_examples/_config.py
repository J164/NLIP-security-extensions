"""Location and loaded configuration for the integration examples."""

from __future__ import annotations

from pathlib import Path

from nlip_sdk.config import load_config

CONFIG_PATH = Path(__file__).with_name("config.toml")
CONFIG = load_config(CONFIG_PATH)
