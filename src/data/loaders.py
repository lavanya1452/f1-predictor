"""Helpers for loading cached JSON responses without contacting the API."""

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Read a JSON file, preserving the response's original data structure."""
    file_path = Path(path)
    try:
        with file_path.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {file_path}: {error}") from error