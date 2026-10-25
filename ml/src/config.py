"""Shared config + path helpers. Loads ml/config/pipeline.yaml once."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "ml" / "config" / "pipeline.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def p(rel: str) -> Path:
    """Resolve a repo-relative path from config to an absolute Path."""
    return ROOT / rel
