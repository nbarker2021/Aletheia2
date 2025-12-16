"""
Structured logging utilities for CQE
"""

import logging
import sys
from typing import Optional


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
):
    """
    Configure logging for CQE application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log format string
        log_file: Optional file path for logging
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)
