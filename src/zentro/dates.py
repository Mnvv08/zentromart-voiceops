"""Deterministic date arithmetic for the agent (LLMs are unreliable at counting business days).

Business days: Monday to Saturday, skipping data/holidays.csv (KB-00.3).
Return windows: calendar days (KB-00.3, RET-2).
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
import re
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def holidays() -> dict[str, str]:
    with open(config.DATA_DIR / "holidays.csv", encoding="utf-8", newline="") as f:
        return {r["date"]: r["name"] for r in csv.DictReader(f)}


def parse(d: str) -> date:
    return date.fromisoformat(str(d).strip()[:10])


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d.isoformat() not in holidays()


def add_business_days(start: str, n: int) -> str:
    d = parse(start)
    left = int(n)
    while left > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            left -= 1
    return d.isoformat()


def add_calendar_days(start: str, n: int) -> str:
    return (parse(start) + timedelta(days=int(n))).isoformat()


def business_days_between(start: str, end: str) -> int:
    """Business days after `start` up to and including `end` (negative if end < start)."""
    a, b = parse(start), parse(end)
    sign = 1
    if b < a:
        a, b, sign = b, a, -1
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if is_business_day(d):
            n += 1
    return sign * n


def pretty(d: str) -> str:
    x = parse(d)
    return f"{x.strftime('%A')} {x.day} {x.strftime('%B %Y')}"


def date_calc(operation: str, start_date: str, days: int = 0, end_date: str = "") -> dict:
    """Tool entry point used by the LLM."""
    try:
        if operation == "add_business_days":
            r = add_business_days(start_date, days)
            return {"result": r, "pretty": pretty(r), "note": "Mon-Sat, holidays skipped"}
        if operation == "add_calendar_days":
            r = add_calendar_days(start_date, days)
            return {"result": r, "pretty": pretty(r)}
        if operation == "business_days_between":
            return {"business_days": business_days_between(start_date, end_date or config.TODAY)}
        if operation == "calendar_days_between":
            return {"calendar_days": (parse(end_date or config.TODAY) - parse(start_date)).days}
        return {"error": f"unknown operation {operation}"}
    except Exception as e:  # bad date from the model
        return {"error": str(e)}


def today_line() -> str:
    return f"Today is {pretty(config.TODAY)} ({config.TODAY}). Holidays: " + ", ".join(
        f"{k} {v}" for k, v in holidays().items())


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def find_dates(text: str) -> list[str]:
    """ISO dates mentioned in text: 2026-10-17, 17 October 2026, 17 Oct, October 17."""
    out = set(re.findall(r"\b(20\d\d-\d\d-\d\d)\b", text))
    low = text.lower()
    mon = "|".join(m[:3] + r"[a-z]*" for m in MONTHS)
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\b(?:,?\s+(20\d\d))?", low):
        out.add(_iso(int(m.group(1)), m.group(2), m.group(3)))
    for m in re.finditer(rf"\b({mon})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d))?", low):
        out.add(_iso(int(m.group(2)), m.group(1), m.group(3)))
    return sorted(d for d in out if d)


def _iso(day: int, mon: str, year: str | None) -> str:
    idx = next((i for i, m in enumerate(MONTHS) if m.startswith(mon[:3])), None)
    if idx is None:
        return ""
    try:
        return date(int(year or 2026), idx + 1, day).isoformat()
    except ValueError:
        return ""


def date_hints(dates: list[str]) -> str:
    """Pre-computed calendar so the model never has to count days itself."""
    lines = []
    for d in dates[:4]:
        bd = ", ".join(f"+{n}={pretty(add_business_days(d, n))}" for n in (1, 2, 3, 5, 7, 10))
        cd = ", ".join(f"+{n}={pretty(add_calendar_days(d, n))}" for n in (7, 10, 15, 30))
        lines.append(f"From {pretty(d)}: business days {bd}. Calendar days {cd}.")
    return "\n".join(lines)
