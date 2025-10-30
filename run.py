"""
Solertia MVP - Restaurant Assistant API
Simple startup script for the API server.
"""
import uvicorn
import logging
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the Solertia MVP API server."""
    try:
        logger.info("🚀 Starting Solertia MVP API Server")
        logger.info("🌐 Server will be available at: http://localhost:8000")
        logger.info("📖 API documentation at: http://localhost:8000/docs")
        logger.info("🏥 Health check at: http://localhost:8000/health")
        
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

