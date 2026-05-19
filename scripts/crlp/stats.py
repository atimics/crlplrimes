"""Numeric summaries and display formatting for experiment reports."""

from __future__ import annotations

import math
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def sample_stdev(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    avg = mean(items)
    return math.sqrt(sum((value - avg) ** 2 for value in items) / (len(items) - 1))


def fmt_pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100.0 * value:.2f}%"


def fmt_delta(value: float) -> str:
    return "-" if math.isnan(value) else f"{100.0 * value:+.2f}pp"


def fmt_float(value: float, digits: int = 4) -> str:
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def percentile(values: Iterable[float], q: float) -> float | None:
    items = sorted(values)
    if not items:
        return None
    if q <= 0:
        return items[0]
    if q >= 1:
        return items[-1]
    idx = q * (len(items) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return items[lo]
    frac = idx - lo
    return items[lo] * (1.0 - frac) + items[hi] * frac

