"""Logging utilities."""

import logging

from colorlog import ColoredFormatter

_LOGGING_INITIALIZED = False


def setup_logging(level: str = "WARNING") -> None:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    logger = logging.getLogger("thinkdepthai")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    color_formatter = ColoredFormatter(
        "%(green)s%(asctime)s%(reset)s[%(blue)s%(name)s%(reset)s] - "
        "%(log_color)s%(levelname)s%(reset)s - %(filename)s:%(lineno)d - %(green)s%(message)s%(reset)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)
    _LOGGING_INITIALIZED = True


def get_logger(name: str, level: int | str = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
