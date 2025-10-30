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
- **Manager Analytics UI**: http://localhost:8000/manager/ui
- **Manager Analytics API**: http://localhost:8000/manager/ask

## 📁 Project Structure

```
Solertia_MVP/
├── src/                    # Source code
│   ├── api/               # API layer
│   │   └── main.py       # FastAPI application
│   ├── core/             # Core business logic
│   │   ├── services/     # Service layer
│   │   │   ├── chat_service.py           # Chat service
│   │   │   ├── reservation_service.py    # Reservation service
│   │   │   ├── menu_service.py           # Menu service
│   │   │   └── manager_analytics_service.py  # Manager Analytics service
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

### Manager Analytics
- `POST /manager/ask` - Process analytics queries using SQL agent
- `GET /manager/plots/{plot_id}` - Get plot visualization by ID
- `GET /manager/map_data` - Get store locations for map visualization
- `GET /manager/ui` - Interactive manager analytics dashboard

### System
- `GET /health` - Health check
- `GET /` - API information

## 🧪 Testing

Test the API using the interactive documentation at http://localhost:8000/docs

### Example Chat Request
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

### Example Manager Analytics Query
```bash
curl -X POST "http://localhost:8000/manager/ask" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Top 10 SKUs por revenue para store_id=3 toda la historia"}'
```

### Manager Analytics UI
Visit http://localhost:8000/manager/ui for an interactive dashboard with:

**Map Features:**
- Interactive map showing all store locations (powered by Plotly.js)
- Visual markers for each restaurant/store
- Click on any store marker to automatically query top SKUs by revenue
- Hover over markers to see store details (name, ID, coordinates)
- Responsive design that adapts to screen size

**Chat Interface:**
- Natural language SQL query interface
- Ask questions like:
  - "Top 10 SKUs por revenue para store_id=3 toda la historia"
  - "Revenue semanal por tienda"
  - "Mejores meseros por propinas"
- Automatic chart generation from query results
- Visual plots displayed inline with responses
- Support for Ctrl+Enter (Cmd+Enter on Mac) to send queries

**Features:**
- Real-time SQL query execution
- Secure read-only database access
- Automatic data visualization
- Store location mapping
- Historical data analysis

## 🔧 Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | Database connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `SUPABASE_PG_CONN` | Supabase PostgreSQL connection (for Manager Analytics) | No* |
| `SUPABASE_SP_CONN` | Supabase connection pooler (preferred for Manager Analytics) | No* |
| `LANGSMITH_API_KEY` | LangSmith API key | No |
| `TZ` | Timezone | No |

\* *Manager Analytics requires a database connection. It will use `SUPABASE_PG_CONN`, `SUPABASE_SP_CONN`, `SUPABASE_DB_URL`, or fall back to `DATABASE_URL`.*

## 🎯 Features

- ✅ **Chat Interface** - Natural conversation with restaurant assistant
- ✅ **Reservation System** - Create and manage reservations
- ✅ **Menu Search** - Search and recommend menu items
- ✅ **Manager Analytics** - SQL-powered analytics agent for managers
  - Natural language to SQL queries
  - Interactive data visualization with charts
  - Store location mapping
  - Read-only queries with security validation
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
- **SQLGlot** - SQL parsing and validation (Manager Analytics)
- **Pandas** - Data analysis (Manager Analytics)
- **Matplotlib** - Chart generation (Manager Analytics)
