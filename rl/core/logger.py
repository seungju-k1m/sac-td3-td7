import logging
import os


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
