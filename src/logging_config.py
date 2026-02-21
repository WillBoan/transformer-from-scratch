import logging
import os
from logging.handlers import RotatingFileHandler
import config as cfg


def setup_logging() -> logging.Logger:
    """Configure root logger: console + rotating file in checkpoint dir."""
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Rotating file handler
    fh_path = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.RUN_NAME}.log")
    fh = RotatingFileHandler(fh_path, maxBytes=10_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Avoid duplicate handlers on repeated setup calls
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
