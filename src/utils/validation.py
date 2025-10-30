"""
Validation utilities for the Solertia MVP.
Provides data validation and sanitization functions.
"""
import re
from datetime import datetime, date, time
from typing import Optional, Dict, Any
from utils.error_handling import ValidationError

def validate_customer_id(customer_id: str) -> str:
    """
    Validate and sanitize customer ID.
    
    Args:
        customer_id: Customer ID to validate
        
    Returns:
        Sanitized customer ID
        
    Note:
        This function accepts both string IDs (for message threads) 
        and numeric IDs (for database customer_id fields)
    """
    if not customer_id or not isinstance(customer_id, str):
        raise ValidationError("Customer ID is required and must be a string")
    
    # Remove any potentially dangerous characters
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', customer_id)
    if len(sanitized) < 1:
        raise ValidationError("Customer ID must be at least 1 characters long")
    
    if len(sanitized) > 100:
        raise ValidationError("Customer ID must be less than 100 characters")
    
    return sanitized

def validate_message(message: str) -> str:
    """Validate and sanitize user message."""
    if not message or not isinstance(message, str):
        raise ValidationError("Message is required and must be a string")
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', message.strip())
    
    if len(sanitized) < 1:
        raise ValidationError("Message cannot be empty")
    
    if len(sanitized) > 2000:
        raise ValidationError("Message must be less than 2000 characters")
    
    return sanitized

def validate_date(date_str: str) -> date:
    """Validate date string in YYYY-MM-DD format."""
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Check if date is not in the past (allowing for today)
        today = date.today()
        if parsed_date < today:
            raise ValidationError("Date cannot be in the past")
        
        # Check if date is not too far in the future (1 year)
        from datetime import timedelta
        max_future = today + timedelta(days=365)
        if parsed_date > max_future:
            raise ValidationError("Date cannot be more than 1 year in the future")
        
        return parsed_date
    except ValueError:
        raise ValidationError("Date must be in YYYY-MM-DD format")

def validate_time(time_str: str) -> time:
    """Validate time string in HH:MM format."""
    try:
        parsed_time = datetime.strptime(time_str, "%H:%M").time()
        return parsed_time
    except ValueError:
        raise ValidationError("Time must be in HH:MM format (24-hour)")

def validate_people_count(people: int) -> int:
    """Validate number of people for reservation."""
    if not isinstance(people, int):
        raise ValidationError("Number of people must be an integer")
    
    if people < 1:
        raise ValidationError("Number of people must be at least 1")
    
    if people > 30:
        raise ValidationError("Number of people cannot exceed 30")
    
    return people

def validate_name(name: str) -> str:
    """Validate and sanitize name."""
    if not name or not isinstance(name, str):
        raise ValidationError("Name is required and must be a string")
    
    # Remove potentially dangerous characters but allow accented characters
    sanitized = re.sub(r'[<>"\']', '', name.strip())
    
    if len(sanitized) < 2:
        raise ValidationError("Name must be at least 2 characters long")
    
    if len(sanitized) > 100:
        raise ValidationError("Name must be less than 100 characters")
    
    return sanitized

def validate_phone(phone: Optional[str]) -> Optional[str]:
    """Validate and sanitize phone number."""
    if not phone:
        return None
    
    if not isinstance(phone, str):
        raise ValidationError("Phone must be a string")
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    if len(digits_only) < 10:
        raise ValidationError("Phone number must have at least 10 digits")
    
    if len(digits_only) > 15:
        raise ValidationError("Phone number cannot have more than 15 digits")
    
    return digits_only

def validate_iso_datetime(iso_str: str) -> datetime:
    """Validate ISO datetime string."""
    try:
        # Handle both with and without timezone info
        if iso_str.endswith('Z'):
            iso_str = iso_str.replace('Z', '+00:00')
        
        parsed = datetime.fromisoformat(iso_str)
        return parsed
    except ValueError:
        raise ValidationError("Invalid ISO datetime format")

def validate_reservation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete reservation data."""
    validated = {}
    
    # Required fields
    validated['name'] = validate_name(data.get('name', ''))
    validated['date'] = validate_date(data.get('date', ''))
    validated['time_start'] = validate_time(data.get('time_start', ''))
    validated['time_end'] = validate_time(data.get('time_end', ''))
    validated['people'] = validate_people_count(data.get('people', 0))
    
    # Optional fields
    if 'phone' in data:
        validated['phone'] = validate_phone(data.get('phone'))
    
    if 'start_iso' in data:
        validated['start_iso'] = validate_iso_datetime(data['start_iso'])
    
    if 'end_iso' in data:
        validated['end_iso'] = validate_iso_datetime(data['end_iso'])
    
    # Validate time logic
    if validated['time_end'] <= validated['time_start']:
        # Allow for overnight reservations (time_end next day)
        pass  # This will be handled by the business logic
    
    return validated
