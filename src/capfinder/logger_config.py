from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from importlib.metadata import version

import loguru
from loguru import logger


def get_version() -> str:
    """Get the version of the app from pyproject.toml."""
    app_version: str = version("capfinder")
    return app_version


# Configure the base logger
log_directory: str = "."


def configure_logger(new_log_directory: str = "", show_location: bool = True) -> str:
    """Configure the logger to log to a file in the specified directory."""
    global log_directory
    log_directory = new_log_directory if new_log_directory else log_directory

    # Ensure log directory exists
    os.makedirs(log_directory, exist_ok=True)

    # Get current date and time
    now: datetime = datetime.now()
    timestamp: str = now.strftime("%Y-%m-%d_%H-%M-%S")
    app_version: str = get_version()

    # Use the timestamp in the log file name
    log_filename: str = f"capfinder_v{app_version}_{timestamp}.log"
    log_filepath: str = os.path.join(log_directory, log_filename)

    # Remove default handler
    logger.remove()

    # Configure logger to log to both file and console with the same format
    log_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | "
    )
    if show_location:
        log_format += (
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        )
    log_format += "<level>{message}</level>"

    logger.add(log_filepath, format=log_format, colorize=True)
    logger.add(sys.stdout, format=log_format, colorize=True)

    return log_filepath


class PrefectHandler(logging.Handler):
    """
    A custom logging handler for Prefect that filters and formats log messages.

    This handler integrates with Loguru, applies custom formatting, and prevents duplicate log messages.
    """

    def __init__(self, loguru_logger: loguru.Logger, show_location: bool) -> None:
        """
        Initialize the PrefectHandler.

        Args:
            loguru_logger (Logger): The Loguru logger instance to use for logging.
            show_location (bool): Whether to show the source location in log messages.
        """
        super().__init__()
        self.loguru_logger = loguru_logger
        self.show_location = show_location
        self.logged_messages: set[str] = set()
        self.prefix_pattern: re.Pattern = re.compile(
            r"(logging:handle:\d+ - )(\w+\.\w+)"
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.

        This method formats the log record, applies custom styling, and logs it using Loguru.
        It also filters out duplicate messages and HTTP status messages.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        try:
            # Filter out HTTP status messages
            if "HTTP Request:" in record.getMessage():
                return

            level: str = record.levelname
            message: str = self.format(record)
            name: str = record.name
            function: str = record.funcName
            line: int = record.lineno

            # Color the part after "logging:handle:XXXX - " cyan
            colored_message: str = self.prefix_pattern.sub(
                r"\1<cyan>\2</cyan>", message
            )

            # Handle progress bar messages
            if "|" in colored_message and (
                "%" in colored_message or "it/s" in colored_message
            ):
                formatted_message: str = f"Progress: {colored_message}"
            else:
                if self.show_location:
                    formatted_message = f"<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {colored_message}"
                else:
                    formatted_message = colored_message

            # Create a unique identifier for this log message
            message_id: str = message

            # Only log if we haven't seen this message before
            if message_id not in self.logged_messages:
                self.logged_messages.add(message_id)
                self.loguru_logger.opt(depth=1, colors=True).log(
                    level, formatted_message
                )
        except Exception:
            self.handleError(record)


def configure_prefect_logging(show_location: bool = True) -> None:
    """
    Configure Prefect logging with custom handler and settings.

    This function sets up a custom PrefectHandler for all Prefect loggers,
    configures the root logger, and adjusts logging levels.

    Args:
        show_location (bool, optional): Whether to show source location in log messages. Defaults to True.
    """
    # Create a single PrefectHandler instance
    handler = PrefectHandler(logger, show_location)

    # Configure the root logger
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Configure all Prefect loggers
    prefect_loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("prefect")
    ]
    for prefect_logger in prefect_loggers:
        prefect_logger.handlers = [handler]
        prefect_logger.propagate = False
        prefect_logger.setLevel(logging.INFO)

    # Disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Export the logger for use in other modules
__all__ = ["logger", "configure_logger", "configure_prefect_logging"]
