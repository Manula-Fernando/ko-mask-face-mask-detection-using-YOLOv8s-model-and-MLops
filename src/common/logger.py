"""
Centralized logging configuration for Face Mask Detection MLOps Pipeline
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

class MLOpsLogger:
    """Centralized logger for all MLOps components"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional structured data"""
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.info(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message with optional structured data"""
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.error(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with optional structured data"""
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.warning(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with optional structured data"""
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        self.logger.debug(message)

# Pre-configured loggers for different components
training_logger = MLOpsLogger("training")
inference_logger = MLOpsLogger("inference")
monitoring_logger = MLOpsLogger("monitoring")
api_logger = MLOpsLogger("api")
pipeline_logger = MLOpsLogger("pipeline")

def get_logger(component: str) -> MLOpsLogger:
    """Get logger for specific component"""
    return MLOpsLogger(component)
