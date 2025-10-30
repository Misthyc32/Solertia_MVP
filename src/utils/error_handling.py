"""
Error handling utilities for the Solertia MVP.
Provides consistent error handling and logging across services.
"""
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from googleapiclient.errors import HttpError as GoogleAPIError

logger = logging.getLogger(__name__)

class SolertiaError(Exception):
    """Base exception for Solertia application errors."""
    def __init__(self, message: str, error_code: str = "SOLERTIA_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DatabaseError(SolertiaError):
    """Database-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)

class CalendarError(SolertiaError):
    """Calendar integration errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CALENDAR_ERROR", details)

class ValidationError(SolertiaError):
    """Data validation errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)

def handle_database_error(error: SQLAlchemyError, context: str = "") -> HTTPException:
    """Handle database errors and return appropriate HTTP response."""
    logger.error(f"Database error in {context}: {str(error)}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "Database operation failed",
            "message": "An internal error occurred while processing your request",
            "context": context
        }
    )

def handle_calendar_error(error: GoogleAPIError, context: str = "") -> HTTPException:
    """Handle Google Calendar API errors."""
    logger.error(f"Calendar API error in {context}: {str(error)}")
    
    if error.resp.status == 404:
        return HTTPException(
            status_code=404,
            detail={
                "error": "Calendar resource not found",
                "message": "The requested calendar event could not be found"
            }
        )
    elif error.resp.status == 403:
        return HTTPException(
            status_code=403,
            detail={
                "error": "Calendar access denied",
                "message": "Insufficient permissions to access calendar"
            }
        )
    else:
        return HTTPException(
            status_code=500,
            detail={
                "error": "Calendar operation failed",
                "message": "An error occurred while processing the calendar request"
            }
        )

def handle_validation_error(error: ValidationError) -> HTTPException:
    """Handle validation errors."""
    logger.warning(f"Validation error: {error.message}")
    return HTTPException(
        status_code=400,
        detail={
            "error": "Validation failed",
            "message": error.message,
            "details": error.details
        }
    )

def handle_generic_error(error: Exception, context: str = "") -> HTTPException:
    """Handle generic errors."""
    logger.error(f"Unexpected error in {context}: {str(error)}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "context": context
        }
    )

def log_service_call(service_name: str, method_name: str, **kwargs):
    """Log service method calls for debugging."""
    logger.info(f"Service call: {service_name}.{method_name} with args: {kwargs}")

def log_service_result(service_name: str, method_name: str, success: bool, result: Any = None):
    """Log service method results."""
    if success:
        logger.info(f"Service call successful: {service_name}.{method_name}")
    else:
        logger.error(f"Service call failed: {service_name}.{method_name}, result: {result}")
