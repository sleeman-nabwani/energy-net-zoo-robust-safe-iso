"""Centralized logging configuration for SafeISO.

This module provides structured logging setup to replace scattered print statements
and debugging outputs throughout the codebase.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "safeiso",
    level: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """Configure and return a logger for SafeISO components.
    
    Args:
        name: Logger name (default: "safeiso")
        level: Logging level override (default: from SAFEISO_LOG_LEVEL env var or INFO)
        format_str: Custom format string (default: standard format with timestamp)
    
    Returns:
        Configured logger instance
    """
    # Get level from environment variable if not specified
    if level is None:
        level = os.environ.get("SAFEISO_LOG_LEVEL", "INFO").upper()
    
    # Default format includes timestamp, level, module, and message
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        module_name: Name of the module requesting the logger
    
    Returns:
        Logger instance for the module
    """
    return setup_logger(f"safeiso.{module_name}")

