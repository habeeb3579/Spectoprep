"""Structured logging for SpectoPrep, built on :mod:`structlog`.

Call :func:`configure_logging` once at application/CLI entry to install the
processor chain, then obtain loggers anywhere with :func:`get_logger`. Library
modules only call :func:`get_logger`; they never configure logging at import
time, so importing SpectoPrep never mutates a host application's logging setup.

Examples
--------
>>> from spectoprep.logging import configure_logging, get_logger
>>> configure_logging(level="INFO")
>>> log = get_logger("example")
>>> log.info("cv_evaluated", rmse=0.12, r2=0.98)  # doctest: +SKIP
"""

from __future__ import annotations

import logging as _stdlib_logging
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool = False, force: bool = False) -> None:
    """Configure structlog and the standard-library logging bridge.

    The ``level`` is always validated. If logging has already been configured
    (for example by an application entry point) this call becomes a no-op unless
    ``force=True``, so a library component instantiated later never overrides an
    explicit application configuration.

    Parameters
    ----------
    level : str, default="INFO"
        Minimum level name ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    json_logs : bool, default=False
        Emit newline-delimited JSON (machine-readable) when True, otherwise a
        colourised, human-readable console renderer.
    force : bool, default=False
        Reconfigure even if logging was already configured.

    Raises
    ------
    ValueError
        If ``level`` is not a recognised logging level name.
    """
    global _CONFIGURED

    numeric_level = getattr(_stdlib_logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Respect an existing configuration (e.g. set by the CLI) unless forced.
    if _CONFIGURED and not force:
        return

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route warnings.warn(...) and stdlib logging through the same pipeline
    # instead of the previous blanket warnings.filterwarnings("ignore").
    _stdlib_logging.basicConfig(level=numeric_level, format="%(message)s")
    _stdlib_logging.captureWarnings(True)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger.

    This never configures logging as a side effect, so importing SpectoPrep
    stays inert. structlog returns a lazy proxy that binds to whatever
    configuration is active at first log call; if :func:`configure_logging`
    was never called, structlog's built-in defaults apply.

    Parameters
    ----------
    name : str, optional
        Logger name, bound as the ``logger`` key on every event.
    """
    return structlog.get_logger(name)
