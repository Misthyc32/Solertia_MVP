"""
FastAPI application for Solertia MVP.
Provides REST API endpoints for chat, reservations, and menu operations.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

# Import services
from src.core.services.chat_service import ChatService
from src.core.services.reservation_service import ReservationService
from src.core.services.menu_service import MenuService

# Import utilities
from src.utils.error_handling import (
    handle_database_error, 
    handle_calendar_error, 
    handle_validation_error,
    handle_generic_error,
    log_service_call,
    log_service_result
)
from src.utils.validation import (
    validate_customer_id,
    validate_message,
    validate_reservation_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Solertia MVP API",
    description="Restaurant assistant API with chat, reservations, and menu services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
chat_service = ChatService()
reservation_service = ReservationService()
menu_service = MenuService()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    customer_id: str = Field(..., description="Customer identifier")
    user_data: Optional[Dict[str, Any]] = Field(None, description="Optional user information")

class ChatResponse(BaseModel):
    response: str
    route: str
    pending_reservation: bool
    pending_update: bool
    reservation_data: Dict[str, Any]
    customer_id: str

class ReservationRequest(BaseModel):
    customer_id: str
    name: str
    date: str  # YYYY-MM-DD
    time_start: str  # HH:MM
    time_end: str  # HH:MM
    party_size: int  # Changed from 'people' to 'party_size'
    start_iso: str  # ISO datetime
    end_iso: str  # ISO datetime
    event_id: Optional[str] = None
    status: str = "confirmed"

class ReservationUpdateRequest(BaseModel):
    event_id: str
    name: Optional[str] = None
    party_size: Optional[int] = None  # Changed from 'people' to 'party_size'
    date: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None

class MenuSearchRequest(BaseModel):
    query: str
    limit: int = 6

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Test database connection
        from src.core.db import SessionLocal
        db = SessionLocal()
        db.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        timestamp=datetime.now().isoformat(),
        services={
            "database": db_status,
            "chat_service": "healthy",
            "reservation_service": "healthy", 
            "menu_service": "healthy"
        }
    )

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return the assistant's response.
    This is the main endpoint for conversation with the restaurant assistant.
    """
    try:
        # Validate inputs
        customer_id = validate_customer_id(request.customer_id)
        message = validate_message(request.message)
        
        logger.info(f"Processing chat message for customer {customer_id}")
        log_service_call("ChatService", "process_message", customer_id=customer_id)
        
        result = chat_service.process_message(
            customer_id=customer_id,
            message=message,
            user_data=request.user_data
        )
        
        log_service_result("ChatService", "process_message", True, result)
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        if "ValidationError" in str(type(e)):
            raise handle_validation_error(e)
        else:
            raise handle_generic_error(e, "chat processing")

@app.get("/chat/{customer_id}/history")
async def get_chat_history(customer_id: str, limit: int = 50):
    """Get conversation history for a customer."""
    try:
        history = chat_service.get_conversation_history(customer_id, limit)
        return {"customer_id": customer_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

# Reservation endpoints
@app.post("/reservations")
async def create_reservation(request: ReservationRequest):
    """Create a new reservation."""
    try:
        logger.info(f"Creating reservation for customer {request.customer_id}")
        
        reservation_data = request.dict()
        result = reservation_service.create_reservation(request.customer_id, reservation_data)
        
        if result["success"]:
            return {"message": "Reservation created successfully", **result}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error creating reservation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating reservation: {str(e)}")

@app.put("/reservations/{event_id}")
async def update_reservation(event_id: str, request: ReservationUpdateRequest):
    """Update an existing reservation."""
    try:
        logger.info(f"Updating reservation {event_id}")
        
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        result = reservation_service.update_reservation(event_id, update_data)
        
        if result["success"]:
            return {"message": "Reservation updated successfully", **result}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error updating reservation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating reservation: {str(e)}")

@app.get("/reservations/{customer_id}")
async def get_reservations(customer_id: str, limit: int = 50):
    """Get reservations for a customer."""
    try:
        reservations = reservation_service.get_reservations(customer_id, limit)
        return {"customer_id": customer_id, "reservations": reservations}
    except Exception as e:
        logger.error(f"Error getting reservations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting reservations: {str(e)}")

@app.get("/reservations/by-event/{event_id}")
async def get_reservation_by_event(event_id: str):
    """Get a reservation by its calendar event ID."""
    try:
        reservation = reservation_service.get_reservation_by_event_id(event_id)
        if reservation:
            return reservation
        else:
            raise HTTPException(status_code=404, detail="Reservation not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reservation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting reservation: {str(e)}")

# Menu endpoints
@app.post("/menu/search")
async def search_menu(request: MenuSearchRequest):
    """Search the menu for items matching the query."""
    try:
        results = menu_service.search_menu(request.query, request.limit)
        return {"query": request.query, "results": results}
    except Exception as e:
        logger.error(f"Error searching menu: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching menu: {str(e)}")

@app.get("/menu/categories")
async def get_menu_categories():
    """Get all available menu categories."""
    try:
        categories = menu_service.get_menu_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error getting menu categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")

@app.post("/menu/recommendations")
async def get_recommendations(preferences: str, limit: int = 3):
    """Get menu recommendations based on preferences."""
    try:
        recommendations = menu_service.get_recommendations(preferences, limit)
        return {"preferences": preferences, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Solertia MVP API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "reservations": "/reservations",
            "menu": "/menu"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
