"""Location and loaded configuration for the shopping assistant."""

from __future__ import annotations

import os
from pathlib import Path

from nlip_sdk.config import load_config

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CONFIG_PATH = Path(os.getenv("SHOP_CONFIG", APP_DIR / "config.toml"))
CONFIG = load_config(CONFIG_PATH)
