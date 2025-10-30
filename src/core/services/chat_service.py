"""
Chat service that handles conversation logic and routing.
Maintains the existing graph/agent logic behind a service layer.
"""
from typing import Dict, Any, List
from src.core.graph import build_app, GlobalState
from src.core.db import SessionLocal, save_message, load_history
from src.core.menu_index import load_menu_vector
import os


class ChatService:
    def __init__(self):
        """Initialize the chat service with vector store and graph."""
        # Load the vector store from the menu
        # Go up to project root: src/core/services -> src/core -> src -> project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        menu_path = os.path.join(project_root, "menu_casona_completo.json")
        self.vector_store = load_menu_vector(menu_path)
        
        # Build the graph/agent
        self.app = build_app(self.vector_store)
    
    def process_message(self, customer_id: str, message: str, user_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user message and return the assistant's response.
        
        Args:
            customer_id: Unique identifier for the customer
            message: User's message
            user_data: Optional user information (phone, name, etc.)
            
        Returns:
            Dictionary containing the response and conversation state
        """
        db = SessionLocal()
        try:
            # Load conversation history
            messages = load_history(db, customer_id)
            
            # Create initial state
            state: GlobalState = {
                "customer_id": customer_id,
                "question": message,
                "messages": messages,
                "context": [],
                "reservation_data": {},
                "answer": "",
                "route": "",
                "pending_reservation": False,
                "pending_update": False,
                "pending_cancel": False,
            }
            
            # Add user data if provided
            if user_data:
                state.update({
                    "phone": user_data.get("phone"),
                    "whatsapp": user_data.get("phone") or user_data.get("whatsapp"),
                    "user_name": user_data.get("name")
                })
            
            # Save user message to database
            save_message(db, customer_id, "user", message)
            
            # Process through the graph/agent
            # Note: LangGraph's MemorySaver uses "thread_id" as the key for checkpointing
            state = self.app.invoke(state, config={"configurable": {"thread_id": customer_id}})
            
            # Save assistant response to database
            save_message(db, customer_id, "assistant", state["answer"])
            
            return {
                "response": state["answer"],
                "route": state.get("route", ""),
                "pending_reservation": state.get("pending_reservation", False),
                "pending_update": state.get("pending_update", False),
                "pending_cancel": state.get("pending_cancel", False),
                "reservation_data": state.get("reservation_data", {}),
                "customer_id": customer_id
            }
            
        finally:
            db.close()
    
    def get_conversation_history(self, customer_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """
        Get conversation history for a customer.
        
        Args:
            customer_id: Unique identifier for the customer
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        db = SessionLocal()
        try:
            return load_history(db, customer_id, limit)
        finally:
            db.close()
