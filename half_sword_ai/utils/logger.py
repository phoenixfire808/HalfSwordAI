"""
Centralized Logging Utilities
Provides consistent logging setup across all modules
"""
import logging
import sys
import time
import os
from pathlib import Path
from typing import Optional
from half_sword_ai.config import config

def _ensure_dir(path: str):
    """Helper to ensure directory exists (avoid circular import)"""
    Path(path).mkdir(parents=True, exist_ok=True)

def setup_logger(
    name: str,
    log_level: Optional[int] = None,
    log_file: Optional[str] = None,
    detailed: Optional[bool] = None
) -> logging.Logger:
    """
    Setup a logger with consistent formatting and handlers
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (defaults to config.DETAILED_LOGGING)
        log_file: Optional log file path (defaults to timestamped file in LOG_PATH)
        detailed: Whether to use detailed logging format (defaults to config.DETAILED_LOGGING)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Determine log level
    if log_level is None:
        log_level = logging.DEBUG if (detailed or config.DETAILED_LOGGING) else logging.INFO
    
    # Determine format
    if detailed is None:
        detailed = config.DETAILED_LOGGING
    
    if detailed:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided or using default)
    if log_file is None:
        # Create timestamped log file in LOG_PATH
        _ensure_dir(config.LOG_PATH)
        log_file = f'{config.LOG_PATH}/{name.split(".")[-1]}_{int(time.time())}.log'
    
    try:
        # Ensure directory exists for log file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            _ensure_dir(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to create file handler: {e}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger already configured, return it
    if logger.handlers:
        return logger
    
    # Otherwise, setup with defaults
    return setup_logger(name)

