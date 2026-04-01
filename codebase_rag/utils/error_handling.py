from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


def log_optional_warning(label: str, error: Exception) -> None:
    """Log a non-fatal warning for an optional subsystem."""
    logger.warning(f"{label} failed: {error}")


def log_and_fallback(
    *,
    label: str,
    error: Exception,
    default: T,
    level: str = "warning",
    include_traceback: bool = False,
) -> T:
    """
    Standardized non-fatal error handler.

    Use when a subsystem failure should degrade gracefully (returning a fallback)
    while still being diagnosable via logs.
    """
    log = logger.opt(exception=error) if include_traceback else logger
    msg = f"{label} failed: {error}"

    match level.lower():
        case "debug":
            log.debug(msg)
        case "info":
            log.info(msg)
        case "error":
            log.error(msg)
        case _:
            log.warning(msg)
    return default


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

