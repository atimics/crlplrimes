"""CLI parsing helpers shared by experiment drivers."""

from __future__ import annotations

import argparse
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root for scripts located under scripts/."""
    return Path(__file__).resolve().parents[2]


def parse_csv(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_int_csv(text: str) -> list[int]:
    return [int(part, 10) for part in parse_csv(text)]


def parse_seeds(text: str) -> list[int]:
    return parse_int_csv(text)

