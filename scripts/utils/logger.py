"""
Extended logging configuration module.

This module sets up logging for the entire application. It configures both console logging and
file logging using a rotating file handler. By adjusting this file, you can enhance logging behavior
without changing logging calls throughout your codebase.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Configuration parameters
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5  # Keep up to 5 backup log files

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Set up the rotating file handler
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, LOG_FILE),
    maxBytes=MAX_BYTES,
    backupCount=BACKUP_COUNT
)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Configure the root logger to use both the console and the file handler
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Create a module-level logger for convenient access
logger = logging.getLogger(__name__)
