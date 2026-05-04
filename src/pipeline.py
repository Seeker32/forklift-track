from __future__ import annotations

from pathlib import Path


def load_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def run(config_path: str | Path) -> None:
    load_config_path(config_path)
