"""Utility functions for loading JSON data from files and zip archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_json_zip(path: Path) -> dict[str, Any]:
    """Load JSON from a zip archive containing a single JSON file.

    Args:
        path: Path to the zip file.

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the zip file does not exist.
        zipfile.BadZipFile: If the file is not a valid zip archive.
        ValueError: If the zip contains no files or multiple files.
    """
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected exactly one file in zip, found {len(names)}")
        with zf.open(names[0]) as f:
            return json.load(f)


def load_json_or_zip(path: Path) -> dict[str, Any]:
    """Load JSON from a file or zip archive, auto-detecting the format.

    If the path ends with .zip, loads from zip. If the path does not exist
    but a .zip version does, loads from the zip version.

    Args:
        path: Path to the JSON file (with or without .zip extension).

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If neither the file nor its .zip version exists.
    """
    if path.suffix == ".zip":
        return load_json_zip(path)

    if path.exists():
        return load_json(path)

    zip_path = Path(str(path) + ".zip")
    if zip_path.exists():
        return load_json_zip(zip_path)

    raise FileNotFoundError(f"Neither {path} nor {zip_path} exists")
