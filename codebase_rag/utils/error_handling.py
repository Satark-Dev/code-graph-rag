from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


def log_optional_warning(label: str, error: Exception) -> None:
    """Log a non-fatal warning for an optional subsystem."""
    logger.warning(f"{label} failed: {error}")


@contextmanager
def optional_section(label: str) -> None:
    """
    Context manager for optional operations.

    Any exception raised inside is logged as a warning with the given label
    and then suppressed.
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001
        log_optional_warning(label, e)


def swallow_optional_error(label: str, fn: Callable[[], T], default: T | None = None) -> T | None:
    """
    Run a callable where failures are non-fatal.

    Errors are logged as warnings with the given label and the default value
    is returned instead.
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        log_optional_warning(label, e)
        return default

