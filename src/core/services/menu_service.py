"""
Menu service that handles menu-related operations.
Provides access to menu information and recommendations.
"""
from typing import List, Dict, Any
from src.core.menu_index import load_menu_vector
from langchain_core.documents import Document
import os


class MenuService:
    def __init__(self):
        """Initialize the menu service with vector store."""
        # Go up to project root: src/core/services -> src/core -> src -> project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        menu_path = os.path.join(project_root, "menu_casona_completo.json")
        self.vector_store = load_menu_vector(menu_path)
    
    def search_menu(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Search the menu for items matching the query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of menu items with metadata
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "category": doc.metadata.get("categoria", "Unknown")
                }
                for doc in results
            ]
        except Exception as e:
            return []
    
    def get_menu_categories(self) -> List[str]:
        """
        Get all available menu categories.
        
        Returns:
            List of category names
        """
        try:
            # This would need to be implemented based on your menu structure
            # For now, return common categories
            return [
                "Entradas",
                "Platos Principales", 
                "Postres",
                "Bebidas",
                "Especialidades"
            ]
        except Exception:
            return []
    
    def get_recommendations(self, preferences: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get menu recommendations based on preferences.
        
        Args:
            preferences: User preferences or dietary requirements
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended menu items
        """
        try:
            results = self.search_menu(preferences, k=limit)
            return results
        except Exception:
            return []
