"""Custom exception classes for FoG detection framework."""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class FoGException(Exception):
    """Base exception class for FoG detection framework."""
    
    def __init__(self, message: str, context: Optional[dict] = None):
        """Initialize with message and optional context."""
        super().__init__(message)
        self.message = message
        self.context = context or {}
        
        # Log the exception with context
        logger.error(f"{self.__class__.__name__}: {message}")
        if self.context:
            logger.error(f"Context: {self.context}")


class DataValidationError(FoGException):
    """Raised when data validation fails."""
    pass


class ConfigurationError(FoGException):
    """Raised when configuration is invalid or missing."""
    pass


class ModelError(FoGException):
    """Raised when model operations fail."""
    pass


class ProcessingError(FoGException):
    """Raised when data processing fails."""
    pass


class CacheError(FoGException):
    """Raised when cache operations fail."""
    pass


class SamplingError(FoGException):
    """Raised when sampling operations fail."""
    pass


class FileSystemError(FoGException):
    """Raised when file system operations fail."""
    pass


def handle_exception(func):
    """Decorator to provide consistent exception handling and logging."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # If it's already a FoGException, re-raise it
            if isinstance(e, FoGException):
                raise
            
            # Convert other exceptions to appropriate FoGException
            func_name = f"{func.__module__}.{func.__name__}"
            context = {
                "function": func_name,
                "args": str(args)[:200] if args else None,
                "kwargs": str(kwargs)[:200] if kwargs else None,
                "original_exception": str(e),
                "exception_type": type(e).__name__
            }
            
            logger.error(f"Unhandled exception in {func_name}: {e}")
            
            # Map common exceptions to FoG exceptions
            if isinstance(e, FileNotFoundError):
                raise FileSystemError(f"File operation failed in {func_name}: {e}", context)
            elif isinstance(e, ValueError):
                raise DataValidationError(f"Data validation failed in {func_name}: {e}", context)
            elif isinstance(e, KeyError):
                raise ConfigurationError(f"Configuration key missing in {func_name}: {e}", context)
            elif isinstance(e, (IOError, OSError)):
                raise FileSystemError(f"I/O operation failed in {func_name}: {e}", context)
            else:
                # Generic wrapping for other exceptions
                raise ProcessingError(f"Processing failed in {func_name}: {e}", context)
    
    return wrapper


def create_error_recovery_strategy(operation_name: str, max_retries: int = 3):
    """Create an error recovery strategy for operations that might fail temporarily."""
    
    def recovery_decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (CacheError, FileSystemError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {operation_name}: {e}. Retrying..."
                        )
                        continue
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {operation_name}")
                        raise
                except Exception as e:
                    # For non-recoverable exceptions, don't retry
                    logger.error(f"Non-recoverable error in {operation_name}: {e}")
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return recovery_decorator


class ErrorContext:
    """Context manager for tracking operation context in errors."""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.logger = logging.getLogger(f"{__name__}.ErrorContext")
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        if self.context:
            self.logger.debug(f"Context: {self.context}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception occurred
            self.logger.error(f"Operation failed: {self.operation}")
            self.logger.error(f"Exception: {exc_type.__name__}: {exc_val}")
            
            # If it's not already a FoGException, wrap it
            if not isinstance(exc_val, FoGException):
                context = {
                    "operation": self.operation,
                    "original_exception": str(exc_val),
                    "exception_type": exc_type.__name__,
                    **self.context
                }
                
                # Choose appropriate exception type based on the original exception
                if issubclass(exc_type, FileNotFoundError):
                    new_exc = FileSystemError(f"File operation failed: {exc_val}", context)
                elif issubclass(exc_type, ValueError):
                    new_exc = DataValidationError(f"Data validation failed: {exc_val}", context)
                elif issubclass(exc_type, KeyError):
                    new_exc = ConfigurationError(f"Configuration error: {exc_val}", context)
                else:
                    new_exc = ProcessingError(f"Processing error: {exc_val}", context)
                
                # Replace the original exception
                raise new_exc from exc_val
        else:
            self.logger.debug(f"Operation completed successfully: {self.operation}")
        
        return False  # Don't suppress exceptions