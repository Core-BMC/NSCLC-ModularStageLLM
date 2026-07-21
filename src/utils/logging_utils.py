"""Logging utilities for TNM staging workflow."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


logger = logging.getLogger(__name__)


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration with file and console handlers.

    Args:
        config: Configuration dictionary containing 'log_file' key

    Returns:
        Configured logger instance
    """
    global logger
    log_file = config.get('log_file', 'tnm_classification.log')
    
    # Create output directory if it doesn't exist
    import os
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Attach handlers to the ROOT logger so logs from ALL modules
    # (src.agents.consensus, src.histology, src.utils.llm_utils, etc.) propagate here.
    # Previously used getLogger(__name__), so per-node/consensus/LLM logs never reached the file.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # INFO default (all app step logs); --verbose raises to DEBUG

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # File handler with UTF-8 encoding
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # Set formatter for both handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set noisy third-party loggers to WARNING to keep the log readable
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio",
                  "langchain", "langchain_core", "langchain_openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Additional settings for Windows environment
    if sys.platform == 'win32':
        # Change Windows console code page to UTF-8
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page (65001)

    return logger

