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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load processed JSON Lines records and identify malformed lines."""
    file_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with file_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {file_path} at line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected an object in {file_path} at line {line_number}"
                    )
                records.append(record)
    except OSError as error:
        raise ValueError(f"Could not read {file_path}: {error}") from error
    return records