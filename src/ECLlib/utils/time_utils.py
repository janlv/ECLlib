"""Date and time related helpers."""
from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timedelta, time as dt_time
from typing import Iterable


def dates_after(date: datetime, datelist: Iterable[datetime]) -> list[datetime]:
    """Return dates in ``datelist`` occurring on or after ``date``."""
    return [d for d in datelist if d >= date]


def day2time(days: float):
    """Convert fractional days to a ``(day, hour, min, sec, msec)`` namedtuple."""
    rest = (days % 1) * 24 * 3600
    values = [int(days)]
    for base in (3600, 60, 1):
        values.append(rest // base)
        rest -= values[-1] * base
    return namedtuple("Time", "day hour min sec msec")(*values, rest)


def float_range(start: float, stop: float, step: float):
    """Yield floating point numbers from ``start`` to ``stop`` with ``step``."""
    while start < stop:
        yield round(start, 10)
        start += step


def date_range(start, stop: float, step: float = 1, fmt: str | None = None):
    """Return a list of dates from ``start`` spanning ``stop`` days."""
    if not isinstance(start, datetime):
        start = datetime(*start)
    dates = (start + timedelta(days=d) for d in float_range(0, stop, step))
    if fmt:
        return [date.strftime(fmt) for date in dates]
    return list(dates)


def date_to_datetime(dates: Iterable[datetime]) -> list[datetime]:
    """Convert :class:`datetime.date` objects to :class:`datetime.datetime`."""
    return [datetime.combine(date, dt_time.min) for date in dates]


def delta_timestring(string1: str, string2: str) -> str:
    """Return the difference between two ``HH:MM:SS`` strings."""
    fmt = "%H:%M:%S"
    delta = datetime.strptime(string1, fmt) - datetime.strptime(string2, fmt)
    return str(timedelta(seconds=int(delta)))


__all__ = [
    "date_range",
    "date_to_datetime",
    "dates_after",
    "day2time",
    "delta_timestring",
    "float_range",
]
