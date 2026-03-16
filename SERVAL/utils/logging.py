"""
SERVAL Logging Utility

Provides configurable logging for SERVAL components.

Usage:
    from SERVAL.utils import get_logger, set_log_level

    logger = get_logger('SERVAL.controller')
    logger.info("Connected to server")
    logger.debug("Request details: %s", data)

    # Change log level at runtime
    set_log_level('DEBUG')
"""

import logging
import sys
from typing import Optional

# Default format for SERVAL logs
DEFAULT_FORMAT = '[%(name)s] %(levelname)s: %(message)s'
DEFAULT_FORMAT_DEBUG = '%(asctime)s [%(name)s] %(levelname)s: %(message)s (%(filename)s:%(lineno)d)'

# Module-level registry of loggers
_loggers: dict[str, logging.Logger] = {}
_handler: Optional[logging.Handler] = None
_current_level: int = logging.INFO


def _get_handler() -> logging.Handler:
    """Get or create the shared stream handler."""
    global _handler
    if _handler is None:
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    return _handler


def get_logger(name: str = 'SERVAL') -> logging.Logger:
    """
    Get a logger instance for the given name.

    Parameters:
        name: Logger name (e.g., 'SERVAL', 'SERVAL.controller')

    Returns:
        logging.Logger: Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(_current_level)
    logger.addHandler(_get_handler())
    logger.propagate = False

    _loggers[name] = logger
    return logger


def set_log_level(level: str | int) -> None:
    """
    Set logging level for all SERVAL loggers.

    Parameters:
        level: Log level - either string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               or logging constant (logging.DEBUG, logging.INFO, etc.)

    Examples:
        set_log_level('DEBUG')
        set_log_level(logging.WARNING)
    """
    global _current_level

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    _current_level = level

    # Update format based on level
    handler = _get_handler()
    if level <= logging.DEBUG:
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT_DEBUG))
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    # Update all existing loggers
    for logger in _loggers.values():
        logger.setLevel(level)


def set_log_format(fmt: str) -> None:
    """
    Set custom log format.

    Parameters:
        fmt: Format string (see logging.Formatter documentation)
    """
    handler = _get_handler()
    handler.setFormatter(logging.Formatter(fmt))
