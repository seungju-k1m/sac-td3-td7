"""Miscellaneous Code."""

import os
import logging
from typing import Any


def convert_dict_as_param(d_str_value: dict[str, Any]) -> dict[str, Any]:
    """Convert dict as param format."""
    param: dict[str, Any] = dict()
    for key, value in d_str_value.items():
        if not isinstance(value, dict):
            param[key] = value
        else:
            param.update(value)
    return param


def setup_logger(path: str, level=logging.DEBUG) -> logging.Logger:
    """Set up logger."""
    logger = logging.getLogger(path)
    logger.setLevel(level)

    if os.path.isfile(path):
        os.remove(path)
    file_handler = logging.FileHandler(path)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


def clamp(x: float, min_x: float, max_x: float) -> float:
    """Clamp x within range."""
    return max(min_x, min(x, max_x))
