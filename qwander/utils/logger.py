"""
This module defines a package-level logger for the `qwander` package.
"""

import logging
import sys
from typing import Optional

__all__ = ["logger", "setup_logging"]

# Package-level logger; defaults to silent until configured
logger = logging.getLogger("qwander")
logger.addHandler(logging.NullHandler())


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure the qwander package logger to emit to stdout.

    Parameters
    ----------
    level : Optional[str]
        Logging level name (e.g. "DEBUG", "INFO", "WARNING").
        Invalid or missing values default to "INFO".
    """
    # Determine log level string and validate
    requested = (level or "info").upper()
    if requested not in logging._nameToLevel:
        # Use default and warn user
        logger.warning("Invalid log level '%s'; defaulting to INFO", requested)
        log_level = logging.INFO
    else:
        log_level = logging._nameToLevel[requested]  # type: ignore

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(levelname)s %(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Reset handlers on the package logger
    pkg_logger = logging.getLogger("qwander")
    pkg_logger.handlers.clear()
    pkg_logger.setLevel(log_level)
    pkg_logger.addHandler(handler)

    # Log configuration success
    pkg_logger.debug("qwander logger configured at level %s",
                     logging.getLevelName(log_level))
