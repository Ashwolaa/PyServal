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
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Default format for SERVAL logs
DEFAULT_FORMAT = '[%(name)s] %(levelname)s: %(message)s'
DEFAULT_FORMAT_DEBUG = '%(asctime)s [%(name)s] %(levelname)s: %(message)s (%(filename)s:%(lineno)d)'

# Module-level registry of loggers — protected by _lock
_lock = threading.Lock()
_loggers: dict[str, logging.Logger] = {}
_handler: Optional[logging.Handler] = None
_current_level: int = logging.INFO
_extra_handlers: list[logging.Handler] = []


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
    with _lock:
        if name in _loggers:
            return _loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(_current_level)
        logger.addHandler(_get_handler())
        for h in _extra_handlers:
            logger.addHandler(h)
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


def add_log_handler(handler: logging.Handler) -> None:
    """
    Attach an external handler to all existing and future SERVAL loggers.

    Use this to forward log records to a GUI widget or file sink without
    replacing the default stream handler.

    Parameters:
        handler: Any logging.Handler instance (e.g. a Qt signal handler)
    """
    with _lock:
        for logger in _loggers.values():
            if handler not in logger.handlers:
                logger.addHandler(handler)
        if handler not in _extra_handlers:
            _extra_handlers.append(handler)


def remove_log_handler(handler: logging.Handler) -> None:
    """Remove a previously added external handler from all SERVAL loggers."""
    with _lock:
        for logger in _loggers.values():
            logger.removeHandler(handler)
        if handler in _extra_handlers:
            _extra_handlers.remove(handler)


def set_log_format(fmt: str) -> None:
    """
    Set custom log format.

    Parameters:
        fmt: Format string (see logging.Formatter documentation)
    """
    handler = _get_handler()
    handler.setFormatter(logging.Formatter(fmt))


# Tracks the active file handler so it can be replaced/removed cleanly.
_file_handler: Optional[RotatingFileHandler] = None


def enable_file_logging(
    path,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """
    Start logging all SERVAL messages to a rotating log file.

    Calling this again with a different path closes the previous file and
    opens the new one.

    Parameters
    ----------
    path : str or Path
        Destination log file path (parent directory is created if needed).
    max_bytes : int
        Maximum file size before rotation (default 10 MB).
    backup_count : int
        Number of rotated backup files to keep (default 3).
    """
    global _file_handler
    disable_file_logging()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _file_handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    _file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT_DEBUG))
    add_log_handler(_file_handler)


def disable_file_logging() -> None:
    """Close and remove the active file log handler (if any)."""
    global _file_handler
    if _file_handler is not None:
        remove_log_handler(_file_handler)
        _file_handler.close()
        _file_handler = None
