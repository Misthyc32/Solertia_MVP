# Solertia MVP - Restaurant Assistant API

A clean, functional restaurant assistant API with chat, reservations, and menu services.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file with your configuration:
```env
DATABASE_URL=sqlite:///./solertia_local.db
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
TZ=America/Monterrey
```

### 3. Start the Server
```bash
python run.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
Solertia_MVP/
├── src/                    # Source code
│   ├── api/               # API layer
│   │   └── main.py       # FastAPI application
│   ├── core/             # Core business logic
│   │   ├── services/     # Service layer
│   │   ├── graph.py      # Conversation graph
│   │   ├── db.py         # Database models
│   │   ├── tools.py      # Calendar tools
│   │   └── config.py     # Configuration
│   └── utils/            # Utilities
├── run.py                # Simple startup script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 📡 API Endpoints

### Chat
- `POST /chat` - Process chat messages
- `GET /chat/{customer_id}/history` - Get conversation history

### Reservations
- `POST /reservations` - Create reservation
- `PUT /reservations/{event_id}` - Update reservation
- `GET /reservations/{customer_id}` - Get user reservations

### Menu
- `POST /menu/search` - Search menu items
- `GET /menu/categories` - Get menu categories

### System
- `GET /health` - Health check
- `GET /` - API information

## 🧪 Testing

Test the API using the interactive documentation at http://localhost:8000/docs

Example chat request:
```json
{
  "message": "Hola, quiero hacer una reservación para 4 personas mañana a las 8pm",
  "customer_id": "user_123",
  "user_data": {
    "phone": "+1234567890",
    "name": "Juan Pérez"
  }
}
```

## 🔧 Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | Database connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key | No |
| `TZ` | Timezone | No |

## 🎯 Features

- ✅ **Chat Interface** - Natural conversation with restaurant assistant
- ✅ **Reservation System** - Create and manage reservations
- ✅ **Menu Search** - Search and recommend menu items
- ✅ **Health Monitoring** - System health checks
- ✅ **Input Validation** - Secure input handling
- ✅ **Error Handling** - Comprehensive error management

## 🚀 Next Steps

1. **Start the server**: `python run.py`
2. **Test the API**: Visit http://localhost:8000/docs
3. **Build UI**: Create frontend interface
4. **Deploy**: Use Docker or cloud platform

## 📝 Development

The API is built with:
- **FastAPI** - Modern, fast web framework
- **SQLAlchemy** - Database ORM
- **LangChain** - AI/LLM integration
- **Pydantic** - Data validation
