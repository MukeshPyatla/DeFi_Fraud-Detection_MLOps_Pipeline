"""
Logging configuration for the DeFi Fraud Detection Pipeline.
"""

import logging
import structlog
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "defi_fraud_detection",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "json"
) -> structlog.BoundLogger:
    """
    Set up structured logging for the application.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Log format (json, console)
    
    Returns:
        Configured structured logger
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message with additional context."""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message with additional context."""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message with additional context."""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message with additional context."""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, exc_info=True, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, exc_info=exc_info, **kwargs)


# Performance monitoring decorator
def log_performance(func):
    """Decorator to log function performance metrics."""
    def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Function executed successfully",
                function_name=func.__name__,
                execution_time=execution_time,
                status="success"
            )
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Function execution failed",
                function_name=func.__name__,
                execution_time=execution_time,
                error=str(e),
                status="error"
            )
            raise
    
    return wrapper


# Audit logging
class AuditLogger:
    """Specialized logger for audit trail events."""
    
    def __init__(self):
        self.logger = structlog.get_logger("audit")
    
    def log_prediction(self, prediction_id: str, model_version: str, 
                      input_data: dict, prediction: dict, confidence: float):
        """Log model prediction for audit trail."""
        self.logger.info(
            "Model prediction made",
            prediction_id=prediction_id,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction,
            confidence=confidence,
            event_type="prediction"
        )
    
    def log_model_change(self, old_version: str, new_version: str, 
                        change_reason: str, performance_metrics: dict):
        """Log model version change."""
        self.logger.info(
            "Model version changed",
            old_version=old_version,
            new_version=new_version,
            change_reason=change_reason,
            performance_metrics=performance_metrics,
            event_type="model_change"
        )
    
    def log_data_access(self, user_id: str, data_type: str, 
                       access_reason: str, data_size: int):
        """Log data access for compliance."""
        self.logger.info(
            "Data accessed",
            user_id=user_id,
            data_type=data_type,
            access_reason=access_reason,
            data_size=data_size,
            event_type="data_access"
        )
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, user_id: str = None):
        """Log security-related events."""
        self.logger.warning(
            "Security event detected",
            event_type=event_type,
            severity=severity,
            description=description,
            user_id=user_id,
            event_type="security"
        )


# Global audit logger instance
audit_logger = AuditLogger() 