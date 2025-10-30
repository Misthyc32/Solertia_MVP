"""
FastAPI application for Solertia MVP.
Provides REST API endpoints for chat, reservations, and menu operations.
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

# Import services
from src.core.services.chat_service import ChatService
from src.core.services.reservation_service import ReservationService
from src.core.services.menu_service import MenuService
from src.core.services.manager_analytics_service import ManagerAnalyticsService

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
manager_analytics_service = ManagerAnalyticsService()

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
    pending_cancel: bool
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

@app.delete("/reservations/{event_id}")
async def cancel_reservation(event_id: str):
    """Cancel an existing reservation."""
    try:
        logger.info(f"Cancelling reservation {event_id}")
        
        result = reservation_service.cancel_reservation(event_id)
        
        if result["success"]:
            return {"message": "Reservation cancelled successfully", **result}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error cancelling reservation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cancelling reservation: {str(e)}")

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

# ========== MANAGER ANALYTICS ENDPOINTS ==========
# Independent analytics agent for managers with SQL query capabilities

class ManagerQueryRequest(BaseModel):
    prompt: str = Field(..., description="Analytics query in natural language")

@app.post("/manager/ask")
async def manager_ask(request: ManagerQueryRequest, http_request: Request):
    """
    Process an analytics query using the SQL agent.
    Returns text response with optional plot_id for visualization.
    """
    try:
        logger.info(f"Processing manager analytics query: {request.prompt}")
        result = manager_analytics_service.query(request.prompt)
        
        # Build plot URL if plot_id exists
        if result.get("plot_id"):
            plot_url = str(http_request.url_for("manager_get_plot", plot_id=result["plot_id"]))
            result["plot_url"] = plot_url
        
        return result
    except Exception as e:
        logger.error(f"Error in manager analytics query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/manager/plots/{plot_id}", name="manager_get_plot")
async def manager_get_plot(plot_id: str):
    """Get a plot image by ID."""
    data = manager_analytics_service.get_plot(plot_id)
    if not data:
        raise HTTPException(status_code=404, detail="Plot not found or expired")
    return Response(content=data, media_type="image/png")

@app.get("/manager/map_data")
async def manager_map_data():
    """Get store locations for map visualization."""
    try:
        data = manager_analytics_service.get_map_data()
        return JSONResponse(data)
    except Exception as e:
        logger.error(f"Error loading map data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading map data: {str(e)}")

@app.get("/manager/ui")
async def manager_ui():
    """Manager Analytics UI - Interactive dashboard with map and SQL chat."""
    return Response(content=MANAGER_UI_HTML, media_type="text/html")

# Manager Analytics UI HTML
MANAGER_UI_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Manager Analytics · Solertia MVP</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root { --gap: 12px; --bg:#0b0d10; --card:#14181d; --txt:#e9eef5; --muted:#9fb0c3; --brand:#46a6ff; }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; background: var(--bg); color: var(--txt); font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans'; }
    .app { display: grid; grid-template-columns: 1fr 380px; grid-auto-rows: 100%; gap: var(--gap); height: 100%; padding: var(--gap); }
    .pane { background: var(--card); border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; }
    .map-header, .chat-header { padding: 10px 14px; border-bottom: 1px solid #202733; display:flex; align-items:center; gap:10px; }
    .map-header h2, .chat-header h2 { font-size: 16px; margin: 0; font-weight: 600; }
    .map-wrap { flex: 1; min-height: 0; }
    #map { width: 100%; height: 100%; }
    .chat { display:flex; flex-direction:column; height:100%; }
    .chat-body { flex:1; overflow:auto; padding: 10px 14px; display:flex; flex-direction:column; gap:10px; }
    .msg { background:#0e1319; border:1px solid #1e2632; padding:10px 12px; border-radius:12px; max-width: 100%; white-space: pre-wrap; }
    .msg.agent { background:#0f1722; border-color:#1d2836; }
    .msg.user  { background:#101820; border-color:#1c242e; }
    .chat-input { border-top: 1px solid #202733; padding:10px; display:flex; gap:8px; }
    .chat-input textarea { resize: none; flex:1; height:74px; background:#0b1118; color:var(--txt); border:1px solid #1f2733; border-radius:10px; padding:10px; outline:none; }
    .btn { background: var(--brand); color: #00233f; border: none; border-radius: 10px; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    .btn:disabled{ opacity:.6; cursor:not-allowed; }
    .tiny { color: var(--muted); font-size: 12px; }
    .pill { padding: 3px 8px; border-radius:999px; background:#0a121b; border:1px solid #1e2733; font-size:12px; color:#9fb0c3; }
    .plot-thumb { width:100%; border-radius:10px; border:1px solid #1f2733; margin-top:8px; }
    @media (max-width: 1024px) {
      .app { grid-template-columns: 1fr; }
      .pane.chat { height: 48vh; }
      .pane.map  { height: 52vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <section class="pane map">
      <div class="map-header">
        <span class="pill">Mapa</span>
        <h2>Restaurantes</h2>
        <span class="tiny" id="map-meta"></span>
      </div>
      <div class="map-wrap"><div id="map"></div></div>
    </section>

    <aside class="pane chat">
      <div class="chat-header">
        <span class="pill">Agente</span>
        <h2>Consultas SQL y Gráficas</h2>
      </div>
      <div class="chat-body" id="chat"></div>
      <div class="chat-input">
        <textarea id="prompt" placeholder="Ej.: Top 10 SKUs por revenue para store_id=3 toda la historia"></textarea>
        <button id="send" class="btn">Enviar</button>
      </div>
    </aside>
  </div>

<script>
const chatEl   = document.getElementById('chat');
const promptEl = document.getElementById('prompt');
const sendBtn  = document.getElementById('send');
const mapMeta  = document.getElementById('map-meta');
const mapEl    = document.getElementById('map');

function addMsg(text, who='agent', plotUrl=null) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  if (plotUrl) {
    const img = document.createElement('img');
    img.className = 'plot-thumb';
    img.src = plotUrl;
    img.alt = 'Gráfica';
    div.appendChild(img);
  }
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function sendPrompt() {
  const q = (promptEl.value || '').trim();
  if (!q) return;
  sendBtn.disabled = true;
  addMsg(q, 'user');
  promptEl.value = '';
  try {
    const res = await fetch('/manager/ask', {
      method: 'POST',
      headers: { 'content-type':'application/json' },
      body: JSON.stringify({ prompt: q })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const text = data.text || '(Sin texto)';
    const plot = data.plot_url || null;
    addMsg(text, 'agent', plot);
  } catch (err) {
    addMsg('Error: ' + err.message, 'agent');
  } finally {
    sendBtn.disabled = false;
  }
}
sendBtn.addEventListener('click', sendPrompt);
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendPrompt();
  }
});

async function sendPromptWithText(q) {
  try {
    sendBtn.disabled = true;
    const res = await fetch('/manager/ask', {
      method: 'POST',
      headers: { 'content-type':'application/json' },
      body: JSON.stringify({ prompt: q })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    addMsg(data.text || '(Sin texto)', 'agent', data.plot_url || null);
  } catch (err) {
    addMsg('Error: ' + err.message, 'agent');
  } finally {
    sendBtn.disabled = false;
  }
}

async function loadMap() {
  try {
    const res = await fetch('/manager/map_data');
    if (!res.ok) throw new Error('No se pudo cargar /manager/map_data');
    const rows = await res.json();
    mapMeta.textContent = rows.length + ' ubicaciones';

    const sizes = rows.map(r => 8 + 24 * ((r.Volumen ?? 0)));
    const trace = {
      type: 'scattermapbox',
      lat: rows.map(r => r.Latitud),
      lon: rows.map(r => r.Longitud),
      text: rows.map(r => r.Store),
      customdata: rows.map(r => r.StoreId),
      hovertemplate:
        '<b>%{text}</b><br>' +
        'Store ID: %{customdata}<br>' +
        'Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>',
      mode: 'markers',
      marker: { size: sizes }
    };

    const lats = rows.map(r => r.Latitud);
    const lons = rows.map(r => r.Longitud);
    const center = {
      lat: lats.reduce((a,b)=>a+b,0)/lats.length,
      lon: lons.reduce((a,b)=>a+b,0)/lons.length
    };

    const layout = {
      dragmode: 'zoom',
      mapbox: { style: 'open-street-map', center, zoom: 5.5 },
      margin: { l:0, r:0, t:0, b:0 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };

    await Plotly.newPlot(mapEl, [trace], layout, {displaylogo:false, responsive:true});
    window.addEventListener('resize', () => Plotly.Plots.resize(mapEl));

    mapEl.on('plotly_click', (ev) => {
      const pt = ev.points?.[0];
      if (!pt) return;
      const storeId = pt.customdata;
      const storeName = pt.text;
      const prompt = `Top 10 SKUs por revenue para store_id=${storeId} usando toda la historia`;
      addMsg(`🗺️ ${storeName} [${storeId}]\n${prompt}`, 'user');
      sendPromptWithText(prompt);
    });
  } catch (e) {
    console.error(e);
    mapMeta.textContent = 'Error cargando mapa';
  }
}
loadMap();
</script>
</body>
</html>
"""

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
            "menu": "/menu",
            "manager": "/manager"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
