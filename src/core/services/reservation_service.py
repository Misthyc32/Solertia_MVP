"""
Reservation service that handles reservation operations.
Provides a clean interface for reservation management.
"""
from typing import Dict, Any, List, Optional
from src.core.db import (
    SessionLocal, 
    create_reservation, 
    find_reservation_by_event_id,
    list_reservations_by_customer_id,
    upsert_user
)
from src.core.tools import reserva_restaurante_tool, update_reservation_tool
import datetime as dt
from zoneinfo import ZoneInfo
from src.core.config import TZ


class ReservationService:
    def __init__(self):
        """Initialize the reservation service."""
        pass
    
    def create_reservation(self, customer_id: str, reservation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new reservation.
        
        Args:
            customer_id: Customer/Thread ID (unique identifier for the customer)
            reservation_data: Reservation details
            
        Returns:
            Dictionary with reservation result
        """
        db = SessionLocal()
        try:
            # Convert customer_id to int for user operations
            try:
                customer_id_int = int(customer_id)
                # Ensure user exists
                upsert_user(
                    db,
                    customer_id=customer_id_int,
                    whatsapp=reservation_data.get("whatsapp"),
                    first_name=reservation_data.get("first_name")
                )
            except (ValueError, TypeError):
                pass  # Skip user creation if customer_id is not numeric
            
            # Create reservation in database
            date_obj = dt.date.fromisoformat(reservation_data["date"])
            time_start = dt.datetime.strptime(reservation_data["time_start"], "%H:%M").time()
            time_end = dt.datetime.strptime(reservation_data["time_end"], "%H:%M").time()
            
            start_iso = dt.datetime.fromisoformat(reservation_data["start_iso"])
            end_iso = dt.datetime.fromisoformat(reservation_data["end_iso"])
            
            reservation = create_reservation(
                db,
                customer_id=customer_id,
                name=reservation_data.get("first_name") or reservation_data.get("name"),
                date=date_obj,
                time_start=time_start,
                time_end=time_end,
                start_iso=start_iso,
                end_iso=end_iso,
                party_size=reservation_data["party_size"],
                calendar_id="solertia.grp@gmail.com",
                event_id=reservation_data.get("event_id"),
                status=reservation_data.get("status", "confirmed")
            )
            
            return {
                "success": True,
                "reservation_id": reservation.reservation_id,
                "event_id": reservation.event_id,
                "status": reservation.status
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()
    
    def update_reservation(self, event_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing reservation.
        
        Args:
            event_id: Calendar event ID
            update_data: Fields to update
            
        Returns:
            Dictionary with update result
        """
        db = SessionLocal()
        try:
            # Find reservation by event ID
            reservation = find_reservation_by_event_id(db, event_id)
            if not reservation:
                return {
                    "success": False,
                    "error": "Reservation not found"
                }
            
            # Update fields if provided
            updated_fields = []
            
            if "first_name" in update_data and update_data["first_name"] != reservation.name:
                reservation.name = update_data["first_name"]
                updated_fields.append("first_name")
            
            if "party_size" in update_data and update_data["party_size"] != reservation.party_size:
                reservation.party_size = update_data["party_size"]
                updated_fields.append("party_size")
            
            if "date" in update_data:
                new_date = dt.date.fromisoformat(update_data["date"])
                if new_date != reservation.date:
                    reservation.date = new_date
                    updated_fields.append("date")
            
            if "time_start" in update_data:
                new_time = dt.datetime.strptime(update_data["time_start"], "%H:%M").time()
                if new_time != reservation.time_start:
                    reservation.time_start = new_time
                    updated_fields.append("time_start")
            
            if "time_end" in update_data:
                new_time = dt.datetime.strptime(update_data["time_end"], "%H:%M").time()
                if new_time != reservation.time_end:
                    reservation.time_end = new_time
                    updated_fields.append("time_end")
            
            if updated_fields:
                reservation.updated_at = dt.datetime.now(ZoneInfo(TZ))
                db.commit()
            
            return {
                "success": True,
                "updated_fields": updated_fields,
                "reservation_id": reservation.reservation_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()
    
    def get_reservations(self, customer_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get reservations for a customer.
        
        Args:
            customer_id: Customer/Thread ID (unique identifier for the customer)
            limit: Maximum number of reservations to return
            
        Returns:
            List of reservation dictionaries
        """
        db = SessionLocal()
        try:
            reservations = list_reservations_by_customer_id(db, customer_id, limit)
            return [
                {
                    "reservation_id": r.reservation_id,
                    "name": r.name,
                    "date": r.date.isoformat() if r.date else None,
                    "time_start": r.time_start.strftime("%H:%M") if r.time_start else None,
                    "time_end": r.time_end.strftime("%H:%M") if r.time_end else None,
                    "party_size": r.party_size,
                    "status": r.status,
                    "event_id": r.event_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None
                }
                for r in reservations
            ]
        finally:
            db.close()
    
    def get_reservation_by_event_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a reservation by its calendar event ID.
        
        Args:
            event_id: Calendar event ID
            
        Returns:
            Reservation dictionary or None
        """
        db = SessionLocal()
        try:
            reservation = find_reservation_by_event_id(db, event_id)
            if not reservation:
                return None
            
            return {
                "reservation_id": reservation.reservation_id,
                "name": reservation.name,
                "date": reservation.date.isoformat() if reservation.date else None,
                "time_start": reservation.time_start.strftime("%H:%M") if reservation.time_start else None,
                "time_end": reservation.time_end.strftime("%H:%M") if reservation.time_end else None,
                "party_size": reservation.party_size,
                "status": reservation.status,
                "event_id": reservation.event_id,
                "created_at": reservation.created_at.isoformat() if reservation.created_at else None,
                "updated_at": reservation.updated_at.isoformat() if reservation.updated_at else None
            }
        finally:
            db.close()
