# MVP Final Solertia

Monorepo del sistema **Solertia** para restaurantes: backend en FastAPI y frontend en React/Vite.

- `backend/` → API de asistente de restaurante (chat, reservaciones, menú, Manager Analytics, integración con Google Calendar y WhatsApp).
- `frontend/` → Panel de administración y CRM con dashboards, mapas y UI para managers.

---

## Estructura

```text
FinalCompletoSolertia/
├── backend/
│   ├── src/
│   │   ├── api/                  # FastAPI (main.py, rutas)
│   │   ├── core/                 # Lógica de negocio
│   │   │   ├── services/         # Servicios (chat, CRM, menú, reservaciones, analytics)
│   │   │   ├── db.py             # Modelos y conexión DB
│   │   │   ├── graph.py          # Grafo de conversación
│   │   │   ├── tools.py          # Herramientas (calendar, etc.)
│   │   │   └── config.py         # Configuración y .env
│   │   └── utils/                # Utilidades y manejo de errores
│   ├── requirements.txt
│   ├── run.py
│   └── README.md                 # Detalle del backend
└── frontend/
    ├── src/
    │   ├── pages/                # Dashboard, Chat, Reservations, Menu, Analytics, CRM
    │   ├── components/           # Layout + componentes shadcn
    │   └── lib/                  # Utilidades
    ├── package.json
    ├── vite.config.ts
    └── README.md                 # Detalle del frontend
```

---

## Requisitos

- **Python 3.12.4**
- **Node 18+** (o versión compatible con Vite)
- Cuenta de **OpenAI** (para el asistente y analytics)
- (Opcional) Base de datos PostgreSQL / Supabase para Manager Analytics

---

## 1. Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Crea un archivo `.env` en `backend/`:

```env
DATABASE_URL=sqlite:///./solertia_local.db

OPENAI_API_KEY=tu_api_key_de_openai
LANGSMITH_API_KEY=tu_api_key_de_langsmith

# Opcionales
TZ=America/Monterrey

# Opcionales para Manager Analytics
SUPABASE_SP_CONN=postgresql://usuario:pass@host:puerto/db
```

Levanta el servidor:

```bash
python run.py
```

Endpoints principales:

- `GET /health`
- `POST /chat`
- `POST /reservations`
- `GET /manager/ui`
- `POST /manager/ask`
- `GET /crm/customers`

Más detalle en [`backend/README.md`](backend/README.md).

---

## 2. Frontend

```bash
cd frontend
npm install
```

Crea un archivo `.env` en `frontend/`:

```env
VITE_API_URL=http://localhost:8000
```

Levanta el frontend:

```bash
npm run dev
```

Abre el navegador en la URL que indique Vite (por defecto `http://localhost:5173`).

---

## Notas de desarrollo

- Los archivos sensibles **no** se suben a Git:
  - `backend/.env`, `backend/credentials`, `backend/token`
  - `frontend/.env`
  - `frontend/node_modules/`
- Si vas a desplegar, crea `.env` propios en el servidor y ajusta:
  - `DATABASE_URL` / `SUPABASE_SP_CONN`
  - `OPENAI_API_KEY`
  - `VITE_API_URL` (URL pública del backend)

---

## TODO / ideas futuras

- Integrar flujo completo de WhatsApp + Twilio + Google Calendar en el README.
- Scripts de `docker-compose` para levantar backend + DB + frontend.
- Tests automatizados para endpoints críticos (reservations, CRM, analytics).
