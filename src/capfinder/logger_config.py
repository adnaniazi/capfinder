"""
This module configures the logger for the capfinder package.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import os
from datetime import datetime
from importlib.metadata import version
from typing import Any

from loguru import logger as base_logger


def get_version() -> Any:
    """Get the version of the app from pyproject.toml.

    Returns:
        app_version (Any): Version of the app.
    """
    app_version = version("capfinder")
    return app_version


# Configure the base logger
logger = base_logger
# Default log directory
log_directory = "."


def configure_logger(new_log_directory: str = "") -> str:
    """Configure the logger to log to a file in the specified directory.

    Args:
        new_log_directory (str): The directory to save the log file in. Defaults to the current directory.

    Returns:
        log_filepath (str): The path to the log file.
    """
    global log_directory
    log_directory = new_log_directory if new_log_directory else log_directory

    # Ensure log directory exists
    os.makedirs(log_directory, exist_ok=True)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    app_version = get_version()

    # Use the timestamp in the log file name
    log_filename = f"capfinder_v{app_version}_{timestamp}.log"

    log_filepath = os.path.join(log_directory, log_filename)

    # Configure logger to log to the file
    logger.add(log_filepath, format="{time} {level} {message}")

    # Now logs will be sent to both the terminal and log_filename
    logger.opt(depth=1).info(f"Started CAPFINDER v{app_version}!")
    return log_filepath
