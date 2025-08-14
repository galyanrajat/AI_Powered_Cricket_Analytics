
import logging
from pathlib import Path

def get_logger(name: str, logs_dir: str = "logs") -> logging.Logger:
    """
    Creates and returns a logger with consistent formatting for all modules.
    Logs go to both console and a file in logs_dir/pipeline.log.
    """
    # Ensure logs directory exists
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        # File handler
        file_handler = logging.FileHandler(Path(logs_dir) / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger
