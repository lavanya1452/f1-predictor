"""Small shared logging setup for command-line and app entry points."""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently configured named logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger