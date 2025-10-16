import os
from dotenv import load_dotenv

# Carga de variables de entorno
load_dotenv(r"C:\Users\cabal\Desktop\Solertia\Solertia_MVP\.env")

# Configuración general
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TZ = "America/Monterrey"

# API Keys
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
