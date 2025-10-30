# restaurant_agent.py
import os, io, base64, re, binascii
from typing import Optional, List
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <- no GUI
import matplotlib.pyplot as plt
from uuid import uuid4
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import sqlglot
from sqlglot.errors import ParseError
import traceback

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse

# ------------- ENV & ENGINE -------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required in .env")

PG_CONN = (
    os.getenv("SUPABASE_PG_CONN")
    or os.getenv("SUPABASE_SP_CONN")
    or os.getenv("SUPABASE_DB_URL")  # optional fallback
)
if not PG_CONN:
    raise RuntimeError(
        "Missing Supabase connection string.\n"
        "Set SUPABASE_PG_CONN or SUPABASE_SP_CONN in your .env file.\n"
        "Example:\n"
        "SUPABASE_SP_CONN=postgresql+psycopg2://postgres:<password>@db.<project>.supabase.co:6543/postgres?sslmode=require"
    )

ENGINE = create_engine(PG_CONN, pool_pre_ping=True, future=True)

ROW_LIMIT_DEFAULT = 50000
STATEMENT_TIMEOUT_MS = 8000

# ---- ALLOWLISTED TABLES (restaurant) ----
ALLOWED_TABLES = {
    'public.menu_items',
    'public.reservation_items',
    'public.reservations',
    'public.stores',
    'public.waiters',
}

# For quoting and regex rewrites (if you want to support unquoted input)
ALLOWED_TABLES_QUOTED = {
    'public.menu_items': 'public."menu_items"',
    'public.reservation_items': 'public."reservation_items"',
    'public.reservations': 'public."reservations"',
    'public.stores': 'public."stores"',
    'public.waiters': 'public."waiters"',
}

# ------------- UTILS -------------
def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _normalize_table_name(tbl_expr) -> str:
    """
    Return fully qualified name if present (catalog.db.table) else table,
    lowercased, without quotes.
    """
    name = (tbl_expr.name or "").replace('"', '')
    db   = (getattr(tbl_expr, "db", None) or "").replace('"', '')
    cat  = (getattr(tbl_expr, "catalog", None) or "").replace('"', '')
    parts = [p for p in [cat, db, name] if p]
    return ".".join(parts).lower() if parts else name.lower()

def _is_allowed_table(name: str) -> bool:
    """
    Accept public.schema.table in ALLOWED_TABLES, or bare table names as public.table.
    """
    nm = name.lower()
    if nm in ALLOWED_TABLES:
        return True
    maybe = f"public.{nm}"
    return maybe in ALLOWED_TABLES

def _is_select_only(sql: str) -> bool:
    """
    Ensure query is SELECT-only, with JOINs allowed, and all tables allowlisted.
    """
    try:
        trees = sqlglot.parse(sql, read="postgres")
    except ParseError:
        return False

    for t in trees:
        # Must contain a Select and no mutating/DDL statements
        if t.find(sqlglot.expressions.Select) is None:
            return False

        banned_node_names = [
            "Insert", "Update", "Delete", "Create", "Drop", "Alter",
            "Truncate", "Grant", "Revoke",
        ]
        for name in banned_node_names:
            Node = getattr(sqlglot.expressions, name, None)
            if Node is not None and t.find(Node) is not None:
                return False

        # Banned functions
        for func in t.find_all(sqlglot.expressions.Func):
            if func.name and func.name.lower() in {"pg_sleep"}:
                return False

        # Every referenced table must be allowlisted
        for tbl in t.find_all(sqlglot.expressions.Table):
            if not _is_allowed_table(_normalize_table_name(tbl)):
                return False

    return True

def _ensure_limit(sql: str, default_limit: int) -> str:
    """
    Garantiza un LIMIT sin depender de parsear 'LIMIT 50000' como AST.
    Si no hay LIMIT, envuelve la consulta en un subquery y agrega LIMIT.
    """
    sql = sql.strip().rstrip(";")

    try:
        trees = sqlglot.parse(sql, read="postgres")
    except ParseError:
        # If unparseable, enforce via wrapper
        return f"SELECT * FROM ({sql}) AS _lim LIMIT {default_limit}"

    # Check any SELECT has LIMIT
    has_limit = False
    for t in trees:
        for sel in t.find_all(sqlglot.expressions.Select):
            if sel.args.get("limit") is not None:
                has_limit = True
                break
        if has_limit:
            break

    if has_limit:
        return sql
    return f"SELECT * FROM ({sql}) AS _lim LIMIT {default_limit}"

def _rewrite_tables_to_quoted(sql: str) -> str:
    """
    Attempt to rewrite FROM/JOIN <table> to quoted public."<table>" variants for all allowlisted tables.
    This is best-effort and safe to skip if the user already quotes correctly.
    """
    s = sql.strip()
    # Direct replacements for fully qualified names
    for bare, quoted in ALLOWED_TABLES_QUOTED.items():
        schema, table = bare.split('.')
        # Replace exact FROM/ JOIN with and without schema
        patterns = [
            (rf'(?i)\bfrom\s+{schema}\.{table}\b', f'FROM {quoted}'),
            (rf'(?i)\bjoin\s+{schema}\.{table}\b', f'JOIN {quoted}'),
            (rf'(?i)\bfrom\s+{table}\b', f'FROM {quoted}'),
            (rf'(?i)\bjoin\s+{table}\b', f'JOIN {quoted}'),
        ]
        for pat, repl in patterns:
            s = re.sub(pat, repl, s)
    return s

def _run_sql_readonly(sql: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
        conn.execute(text(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'"))
        return pd.read_sql(text(sql), conn)

# ------------- TOOLS -------------
class SQLArgs(BaseModel):
    sql: str = Field(..., description="A full SELECT (Postgres). May JOIN allowlisted tables only.")

def run_sql(sql: str) -> str:
    sql = _rewrite_tables_to_quoted(sql.strip().rstrip(";"))
    if not _is_select_only(sql):
        return "SQL rejected: only read-only SELECTs on allowlisted tables are allowed."

    sql2 = _ensure_limit(sql, ROW_LIMIT_DEFAULT)
    try:
        df = _run_sql_readonly(sql2)
    except Exception as e:
        return f"SQL error: {e}"

    meta = f"rows={len(df)}, cols={list(df.columns)}"
    preview_md = df.head(10).to_markdown(index=False)
    preview_json = df.head(10).to_json(orient="records")
    return f"{meta}\n\nPREVIEW:\n{preview_md}\n\nDF_JSON_HEAD:\n{preview_json}"

RunSQLTool = StructuredTool.from_function(
    func=run_sql,
    name="run_sql",
    description="Execute a validated read-only SELECT on Supabase/Postgres and return a preview.",
    args_schema=SQLArgs,
)

class SchemaArgs(BaseModel):
    pass

def get_schema() -> str:
    return (
        "Tables (allowlist):\n"
        'public."menu_items" (\n'
        "  sku TEXT PRIMARY KEY,\n"
        "  name TEXT, description TEXT, price NUMERIC, category TEXT,\n"
        "  is_active BOOLEAN, margen_ganancia NUMERIC\n"
        ")\n\n"
        'public."reservation_items" (\n'
        "  reservation_id INTEGER, sku TEXT, quantity INTEGER, price_at_visit NUMERIC\n"
        ")\n\n"
        'public."reservations" (\n'
        "  reservation_id INTEGER PRIMARY KEY,\n"
        "  customer_id BIGINT, store_id INTEGER,\n"
        "  reservation_time TEXT,   -- dd/mm/yyyy in current sample\n"
        "  party_size INTEGER, status TEXT, special_requests TEXT, waiter_id INTEGER,\n"
        "  table_num INTEGER, tip NUMERIC, total_ticket NUMERIC, payment_method TEXT,\n"
        "  name TEXT, date DATE, time_start TIME, time_end TIME,\n"
        "  start_iso TIMESTAMPTZ, end_iso TIMESTAMPTZ, calendar_id TEXT, event_id TEXT,\n"
        "  created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ\n"
        ")\n\n"
        'public."stores" (\n'
        "  store_id INTEGER PRIMARY KEY, store_name TEXT, location TEXT,\n"
        "  latitude DOUBLE PRECISION, longitude DOUBLE PRECISION\n"
        ")\n\n"
        'public."waiters" (\n'
        "  waiter_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, store_id INTEGER\n"
        ")\n\n"
        "Business notes:\n"
        "- revenue (line) = reservation_items.quantity * reservation_items.price_at_visit.\n"
        "- menu_price = menu_items.price (current list price); margin% = menu_items.margen_ganancia.\n"
        "- Approx earnings = SUM(revenue * COALESCE(menu_items.margen_ganancia, 0)).\n"
        "- Tip rate = tip / NULLIF(total_ticket,0) at the reservation level.\n"
        "- Date handling: prefer reservations.date; if NULL, use to_date(reservation_time,'DD/MM/YYYY').\n"
        "- Store filter: reservations.store_id.\n"
        "- Waiter performance: group by waiters.waiter_id (join via reservations.waiter_id).\n"
        "- Weekly grouping: date_trunc('week', <date_col>) as wk.\n"
    )

SchemaTool = StructuredTool.from_function(
    func=get_schema,
    name="get_schema",
    description="Return the available tables/columns so you can write SQL.",
    args_schema=SchemaArgs,
)

# -- Lightweight plot cache
PLOT_CACHE: dict[str, bytes] = {}
def _cache_put(pid: str, data: bytes) -> str:
    PLOT_CACHE[pid] = data
    return pid

class PlotArgs(BaseModel):
    data_sql: str = Field(..., description="SELECT returning an x column plus one or more numeric y columns.")
    x: str = Field(..., description="x-axis column (e.g., date or wk).")
    y_cols: List[str] = Field(..., description="One or more numeric columns to plot.")
    title: Optional[str] = Field(None, description="Optional chart title.")

def plot_from_sql(data_sql: str, x: str, y_cols: List[str], title: Optional[str] = None) -> str:
    data_sql = _rewrite_tables_to_quoted(data_sql.strip().rstrip(";"))
    if not _is_select_only(data_sql):
        return "SQL rejected in plot: only read-only SELECTs on allowlisted tables."

    # Limita filas para que no se tarde
    data_sql2 = _ensure_limit(data_sql, 5000)

    try:
        df = _run_sql_readonly(data_sql2)
    except Exception as e:
        return f"SQL error: {e}"

    if x not in df.columns or any(col not in df.columns for col in y_cols):
        return f"Missing columns. Available: {list(df.columns)}"

    # Ordena por x si aplica
    try:
        df = df.sort_values(by=[x])
    except Exception:
        pass

    # Downsample si hay demasiados puntos
    if len(df) > 3000:
        step = max(1, len(df) // 1500)
        df = df.iloc[::step].copy()

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for col in y_cols:
        ax.plot(df[x].values, df[col].values, label=col)
    ax.set_xlabel(x); ax.set_ylabel("value")
    ax.set_title(title or "Chart")
    ax.legend(loc="best")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)  # no bbox_inches="tight" to keep size predictable
    plt.close(fig)
    buf.seek(0)

    pid = str(uuid4())
    _cache_put(pid, buf.getvalue())
    return f"PLOT_ID:{pid}"

PlotTool = StructuredTool.from_function(
    func=plot_from_sql,
    name="plot_from_sql",
    description="Run a read-only SELECT, then plot y columns vs x. Returns PLOT_ID:<uuid>.",
    args_schema=PlotArgs,
)

# ------------- AGENT -------------
SYSTEM = """You are the Restaurant Data Agent. You can write your own SQL.

Workflow:
1) Call get_schema if you need column references.
2) Plan briefly, then call run_sql with a single read-only SELECT that may JOIN any allowlisted table.
3) For charts, call plot_from_sql with tidy columns (x + numeric y).

Rules:
- Read-only SELECTs only. No DML/DDL. No pg_sleep.
- Allowed tables: public."menu_items", public."reservation_items", public."reservations", public."stores", public."waiters".
- revenue (line) = quantity * price_at_visit.
- earnings (approx) = SUM(revenue * COALESCE(menu_items.margen_ganancia, 0)).
- tip_rate = tip / NULLIF(total_ticket, 0).
- Dates: prefer reservations.date; else use to_date(reservation_time, 'DD/MM/YYYY').
- Store filter: reservations.store_id = <id>.
- Weekly: date_trunc('week', <date_col>) AS wk; GROUP BY wk ORDER BY wk.
- Keep answers concise. When returning a plot, output EXACTLY:
  PLOT_ID:<uuid>
"""

def build_agent(model_name: str = "gpt-4o-mini") -> AgentExecutor:
    llm = ChatOpenAI(model=model_name, temperature=0)
    tools = [SchemaTool, RunSQLTool, PlotTool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                 # helpful to observe chain of tools
        max_iterations=8,
        early_stopping_method="generate",
    )

AGENT = build_agent()

# ------------- FASTAPI -------------
app = FastAPI(title="Restaurant Agent · Map + Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class AskPayload(BaseModel):
    prompt: str

@app.post("/ask")
def ask(p: AskPayload, request: Request):
    out = AGENT.invoke({"input": p.prompt})["output"]

    # Prefer PLOT_ID
    m_id = re.search(r"PLOT_ID:\s*([0-9a-fA-F-]{36})", out, flags=re.I | re.S)
    if m_id:
        text_part = out[:m_id.start()].strip()
        pid = m_id.group(1)
        url = str(request.url_for("get_plot", plot_id=pid))
        return {"text": text_part, "plot_url": url, "plot_id": pid}

    # Fallback: if base64 was returned (shouldn't), try to render it anyway
    m_b64 = re.search(r"PLOT_BASE64_PNG:\s*([A-Za-z0-9+/=\r\n]+)", out, flags=re.S)
    if m_b64:
        text_part = out[:m_b64.start()].strip()
        raw = m_b64.group(1)
        cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
        try:
            img = base64.b64decode(cleaned, validate=True)
        except binascii.Error:
            img = None

        if img and img.startswith(b"\x89PNG\r\n\x1a\n"):
            pid = str(uuid4())
            PLOT_CACHE[pid] = img
            url = str(request.url_for("get_plot", plot_id=pid))
            return {"text": text_part, "plot_url": url, "plot_id": pid}

        return {"text": text_part}

    # No plot
    return {"text": out}

@app.get("/plots/{plot_id}", name="get_plot")
def get_plot(plot_id: str):
    data = PLOT_CACHE.get(plot_id)
    if not data:
        raise HTTPException(status_code=404, detail="Plot not found or expired")
    return Response(content=data, media_type="image/png")

@app.get("/db_check")
def db_check():
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
            row = conn.execute(text("select current_user, current_database(), inet_server_addr(), inet_server_port()")).fetchone()
        return {"ok": True, "db": list(row)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm_check")
def llm_check():
    try:
        _ = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke("ping")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- MAP DATA from DB (stores table) --------
@app.get("/map_data")
def map_data():
    """
    Reads store locations from public.stores (id, name, lat, lon) for the UI map.
    """
    try:
        sql = """
        SELECT store_id, store_name, location, latitude AS lat, longitude AS lon
        FROM public."stores"
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY store_id
        """
        df = _run_sql_readonly(_ensure_limit(sql, 10000))
        # Normalize columns to the UI expectations
        out = []
        for _, r in df.iterrows():
            out.append({
                "Store": r.get("store_name"),
                "StoreId": int(r.get("store_id")) if pd.notna(r.get("store_id")) else None,
                "Location": r.get("location"),
                "Latitud": float(r.get("lat")),
                "Longitud": float(r.get("lon")),
                "Volumen": 0.5,  # placeholder (bubble size); adjust if you have a metric
            })
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map data: {e}")

HTML_UI = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Restaurant Agent · Mapa + Chat</title>
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
    const res = await fetch('/ask', {
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

// Helper para enviar texto directo (lo usa el click del mapa)
async function sendPromptWithText(q) {
  try {
    sendBtn.disabled = true;
    const res = await fetch('/ask', {
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

// -------- MAPA (Plotly JS con OpenStreetMap) --------
async function loadMap() {
  try {
    const res = await fetch('/map_data');
    if (!res.ok) throw new Error('No se pudo cargar /map_data');
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

    // Click en marcador => pregunta automática al agente
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

@app.get("/ui", name="ui")
def ui():
    return Response(content=HTML_UI, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
