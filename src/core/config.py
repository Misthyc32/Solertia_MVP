import os
from dotenv import load_dotenv

# Carga de variables de entorno
load_dotenv()

# Configuración general
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TZ = "America/Monterrey"

# API Keys
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
