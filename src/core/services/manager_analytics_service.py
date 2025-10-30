"""
Manager Analytics Service - Independent SQL agent for managers.
Provides data analysis capabilities through natural language queries.
"""
import os
import io
import re
import binascii
from typing import Optional, List, Dict
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # No GUI backend
import matplotlib.pyplot as plt
from uuid import uuid4
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import sqlglot
from sqlglot.errors import ParseError

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.config import OPENAI_API_KEY

# Load environment
load_dotenv()

# Database connection - using same as main DB but with read-only access
PG_CONN = (
    os.getenv("SUPABASE_PG_CONN")
    or os.getenv("SUPABASE_SP_CONN")
    or os.getenv("SUPABASE_DB_URL")
    or os.getenv("DATABASE_URL")
)

if not PG_CONN:
    raise RuntimeError(
        "Missing database connection string for Manager Analytics.\n"
        "Set SUPABASE_PG_CONN, SUPABASE_SP_CONN, or DATABASE_URL in .env file."
    )

# Create engine with read-only optimizations
ENGINE = create_engine(PG_CONN, pool_pre_ping=True, future=True)

# Configuration
ROW_LIMIT_DEFAULT = 50000
STATEMENT_TIMEOUT_MS = 8000

# Allowed tables for SQL queries
ALLOWED_TABLES = {
    'public.menu_items',
    'public.reservation_items',
    'public.reservations',
    'public.stores',
    'public.waiters',
}

ALLOWED_TABLES_QUOTED = {
    'public.menu_items': 'public."menu_items"',
    'public.reservation_items': 'public."reservation_items"',
    'public.reservations': 'public."reservations"',
    'public.stores': 'public."stores"',
    'public.waiters': 'public."waiters"',
}

# Plot cache (in-memory, scoped to service instance)
PLOT_CACHE: Dict[str, bytes] = {}


# ------------- UTILITY FUNCTIONS -------------

def _normalize_table_name(tbl_expr) -> str:
    """Return fully qualified name, lowercased, without quotes."""
    name = (tbl_expr.name or "").replace('"', '')
    db = (getattr(tbl_expr, "db", None) or "").replace('"', '')
    cat = (getattr(tbl_expr, "catalog", None) or "").replace('"', '')
    parts = [p for p in [cat, db, name] if p]
    return ".".join(parts).lower() if parts else name.lower()


def _is_allowed_table(name: str) -> bool:
    """Check if table is in allowlist."""
    nm = name.lower()
    if nm in ALLOWED_TABLES:
        return True
    maybe = f"public.{nm}"
    return maybe in ALLOWED_TABLES


def _is_select_only(sql: str) -> bool:
    """Ensure query is SELECT-only with allowlisted tables."""
    try:
        trees = sqlglot.parse(sql, read="postgres")
    except ParseError:
        return False

    for t in trees:
        # Must contain a Select
        if t.find(sqlglot.expressions.Select) is None:
            return False

        # No mutating/DDL statements
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

        # All tables must be allowlisted
        for tbl in t.find_all(sqlglot.expressions.Table):
            if not _is_allowed_table(_normalize_table_name(tbl)):
                return False

    return True


def _ensure_limit(sql: str, default_limit: int) -> str:
    """Ensure SQL has a LIMIT clause."""
    sql = sql.strip().rstrip(";")

    try:
        trees = sqlglot.parse(sql, read="postgres")
    except ParseError:
        return f"SELECT * FROM ({sql}) AS _lim LIMIT {default_limit}"

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
    """Rewrite table names to quoted variants."""
    s = sql.strip()
    for bare, quoted in ALLOWED_TABLES_QUOTED.items():
        schema, table = bare.split('.')
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
    """Execute read-only SQL and return DataFrame."""
    with ENGINE.connect() as conn:
        conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
        conn.execute(text(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'"))
        return pd.read_sql(text(sql), conn)


# ------------- TOOLS -------------

class SQLArgs(BaseModel):
    sql: str = Field(..., description="A full SELECT (Postgres). May JOIN allowlisted tables only.")


def run_sql(sql: str) -> str:
    """Execute validated read-only SELECT and return preview."""
    sql = _rewrite_tables_to_quoted(sql.strip().rstrip(";"))
    if not _is_select_only(sql):
        return "SQL rejected: only read-only SELECTs on allowlisted tables are allowed."

    sql2 = _ensure_limit(sql, ROW_LIMIT_DEFAULT)
    try:
        df = _run_sql_readonly(sql2)
    except Exception as e:
        return f"SQL error: {e}"

    # Check if all date/time columns are NULL (common issue)
    date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'month', 'week', 'day', 'time'])]
    if date_columns:
        null_counts = {}
        for col in date_columns:
            null_count = df[col].isna().sum()
            total = len(df)
            null_counts[col] = f"{null_count}/{total} NULL"
            if null_count == total and total > 0:
                # All values are NULL - suggest using alternative date column
                suggestions = []
                if 'date' in sql.lower():
                    suggestions.append("Try using COALESCE(r.date, r.created_at, to_date(r.reservation_time, 'DD/MM/YYYY')) instead of just r.date")
                    suggestions.append("Or use created_at if date is NULL: date_trunc('month', COALESCE(r.date, r.created_at))")
                return (
                    f"WARNING: All values in '{col}' are NULL ({null_count}/{total} rows). "
                    f"This means the date column used has no values. "
                    f"{' '.join(suggestions)}. "
                    f"\n\nCurrent result: {len(df)} row(s) with NULL dates.\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Null counts: {null_counts}"
                )

    meta = f"rows={len(df)}, cols={list(df.columns)}"
    preview_md = df.head(10).to_markdown(index=False)
    preview_json = df.head(10).to_json(orient="records")
    
    # Add warning if single row with NULL date for time-series queries
    if len(df) == 1 and date_columns:
        date_col = date_columns[0]
        if df[date_col].iloc[0] is None or pd.isna(df[date_col].iloc[0]):
            warning = f"\n\n⚠️ WARNING: Only 1 row returned with NULL {date_col}. For historical charts, you need multiple data points. Try using COALESCE(r.date, r.created_at, to_date(r.reservation_time, 'DD/MM/YYYY')) AS date_col in your GROUP BY."
            return f"{meta}\n\nPREVIEW:\n{preview_md}\n\nDF_JSON_HEAD:\n{preview_json}{warning}"
    
    return f"{meta}\n\nPREVIEW:\n{preview_md}\n\nDF_JSON_HEAD:\n{preview_json}"


RunSQLTool = StructuredTool.from_function(
    func=run_sql,
    name="run_sql",
    description="Execute a validated read-only SELECT on Postgres and return a preview.",
    args_schema=SQLArgs,
)


class SchemaArgs(BaseModel):
    pass


def get_schema() -> str:
    """Return database schema information."""
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
        "- Date handling: ALWAYS use COALESCE for dates: COALESCE(reservations.date, reservations.created_at, to_date(reservations.reservation_time,'DD/MM/YYYY')) AS date_col.\n"
        "  * reservations.date may be NULL, so fallback to created_at or parse reservation_time.\n"
        "  * Example: date_trunc('month', COALESCE(r.date, r.created_at, to_date(r.reservation_time, 'DD/MM/YYYY'))) AS month\n"
        "- Store filter: reservations.store_id.\n"
        "- Waiter performance: group by waiters.waiter_id (join via reservations.waiter_id).\n"
        "- Weekly grouping: date_trunc('week', COALESCE(date_col, created_at)) AS wk.\n"
    )


SchemaTool = StructuredTool.from_function(
    func=get_schema,
    name="get_schema",
    description="Return the available tables/columns so you can write SQL.",
    args_schema=SchemaArgs,
)


class PlotArgs(BaseModel):
    data_sql: str = Field(..., description="SELECT returning an x column plus one or more numeric y columns.")
    x: str = Field(..., description="x-axis column (e.g., date or wk).")
    y_cols: List[str] = Field(..., description="One or more numeric columns to plot.")
    title: Optional[str] = Field(None, description="Optional chart title.")


def plot_from_sql(data_sql: str, x: str, y_cols: List[str], title: Optional[str] = None, plot_cache: Dict[str, bytes] = None) -> str:
    """Generate a plot from SQL query and return plot ID."""
    if plot_cache is None:
        plot_cache = PLOT_CACHE

    data_sql = _rewrite_tables_to_quoted(data_sql.strip().rstrip(";"))
    if not _is_select_only(data_sql):
        return "SQL rejected in plot: only read-only SELECTs on allowlisted tables."

    data_sql2 = _ensure_limit(data_sql, 5000)

    try:
        df = _run_sql_readonly(data_sql2)
    except Exception as e:
        return f"SQL error: {e}"

    if x not in df.columns or any(col not in df.columns for col in y_cols):
        return f"Missing columns. Available: {list(df.columns)}"

    # Validate we have enough data points for a meaningful chart
    if len(df) < 2:
        return (
            f"Query returned only {len(df)} row(s). For historical charts, you need multiple data points over time. "
            f"Make sure your SQL includes a GROUP BY with a date/time column (like date, wk, month, etc.) "
            f"and ORDER BY that column. Example: SELECT date_trunc('week', date) AS wk, SUM(revenue) AS total "
            f"FROM ... GROUP BY wk ORDER BY wk"
        )

    # Sort by x if possible
    try:
        df = df.sort_values(by=[x])
    except Exception:
        pass

    # Convert x column to string if it's not numeric or datetime (for better display)
    try:
        # Try to convert dates if they're strings
        if df[x].dtype == 'object':
            try:
                df[x] = pd.to_datetime(df[x])
            except (ValueError, TypeError):
                pass
    except Exception:
        pass

    # Downsample if too many points
    if len(df) > 3000:
        step = max(1, len(df) // 1500)
        df = df.iloc[::step].copy()

    # Validate y columns are numeric
    for col in y_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                return f"Column '{col}' is not numeric and cannot be converted. Available numeric columns: {[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]}"

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for col in y_cols:
        # Filter out NaN values
        valid_mask = pd.notna(df[x]) & pd.notna(df[col])
        if valid_mask.sum() < 1:
            continue
        ax.plot(df[valid_mask][x].values, df[valid_mask][col].values, label=col, marker='o', markersize=3)
    ax.set_xlabel(x)
    ax.set_ylabel("value")
    ax.set_title(title or "Chart")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if they're dates or strings
    if df[x].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[x]):
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    pid = str(uuid4())
    plot_cache[pid] = buf.getvalue()
    return f"PLOT_ID:{pid}"


# ------------- AGENT -------------

SYSTEM_PROMPT = """You are the Restaurant Data Analytics Agent for managers. You can write your own SQL.

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
- Dates: ALWAYS use COALESCE to handle NULL dates. Use: COALESCE(reservations.date, reservations.created_at, to_date(reservations.reservation_time, 'DD/MM/YYYY')) AS date_col
- If reservations.date is NULL, fallback to created_at or parse reservation_time.

CRITICAL FOR HISTORICAL CHARTS:
- When creating historical/time-series charts, you MUST include a time dimension in your SELECT.
- ALWAYS use GROUP BY with a date/time column (date, week, month, etc.) for historical data.
- NEVER return just a single aggregated total for charts - you need multiple data points over time.
- Examples for historical queries (with NULL handling):
  * Daily: SELECT DATE(COALESCE(r.date, r.created_at, to_date(r.reservation_time, 'DD/MM/YYYY'))) AS day, SUM(ri.quantity * ri.price_at_visit) AS total FROM public.reservations r JOIN public.reservation_items ri ON r.reservation_id = ri.reservation_id GROUP BY day ORDER BY day
  * Weekly: SELECT date_trunc('week', COALESCE(r.date, r.created_at, to_date(r.reservation_time, 'DD/MM/YYYY'))) AS wk, SUM(ri.quantity * ri.price_at_visit) AS total FROM public.reservations r JOIN public.reservation_items ri ON r.reservation_id = ri.reservation_id GROUP BY wk ORDER BY wk
  * Monthly: SELECT date_trunc('month', COALESCE(r.date, r.created_at, to_date(r.reservation_time, 'DD/MM/YYYY'))) AS month, SUM(ri.quantity * ri.price_at_visit) AS total FROM public.reservations r JOIN public.reservation_items ri ON r.reservation_id = ri.reservation_id GROUP BY month ORDER BY month
- The x-axis column for plot_from_sql MUST be the time dimension (day, wk, month, date, etc.)
- The y-axis columns must be numeric aggregations (SUM, AVG, COUNT, etc.)
- Always ORDER BY the time column to ensure chronological order.

General rules:
- Store filter: reservations.store_id = <id>.
- Keep answers concise. When returning a plot, output EXACTLY:
  PLOT_ID:<uuid>
"""


class ManagerAnalyticsService:
    """Service for manager analytics with independent SQL agent."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the manager analytics service."""
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required in .env")

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Create plot cache for this instance
        self.plot_cache: Dict[str, bytes] = {}
        
        # Create plot tool with instance-specific cache
        def plot_with_cache(data_sql: str, x: str, y_cols: List[str], title: Optional[str] = None) -> str:
            return plot_from_sql(data_sql, x, y_cols, title, self.plot_cache)

        PlotTool = StructuredTool.from_function(
            func=plot_with_cache,
            name="plot_from_sql",
            description=(
                "Run a read-only SELECT query, then plot y columns vs x. Returns PLOT_ID:<uuid>.\n"
                "CRITICAL: For historical/time-series charts, the SQL MUST include:\n"
                "1. A time dimension column (date, wk, month, etc.) in the SELECT\n"
                "2. GROUP BY with that time column\n"
                "3. ORDER BY that time column\n"
                "4. Multiple rows (not just one aggregated total)\n"
                "Example: SELECT date_trunc('week', date) AS wk, SUM(revenue) AS total FROM ... GROUP BY wk ORDER BY wk\n"
                "Then use: x='wk', y_cols=['total']"
            ),
            args_schema=PlotArgs,
        )

        self.tools = [SchemaTool, RunSQLTool, PlotTool]

        # Build agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
        )

    def query(self, prompt: str) -> Dict:
        """
        Process an analytics query and return response.

        Args:
            prompt: Natural language query

        Returns:
            Dictionary with text response and optional plot_url/plot_id
        """
        try:
            result = self.agent_executor.invoke({"input": prompt})
            out = result["output"]

            # Check for PLOT_ID
            m_id = re.search(r"PLOT_ID:\s*([0-9a-fA-F-]{36})", out, flags=re.I | re.S)
            if m_id:
                text_part = out[:m_id.start()].strip()
                pid = m_id.group(1)
                return {
                    "text": text_part,
                    "plot_id": pid,
                }

            # Fallback: base64 (shouldn't happen with new tool)
            m_b64 = re.search(r"PLOT_BASE64_PNG:\s*([A-Za-z0-9+/=\r\n]+)", out, flags=re.S)
            if m_b64:
                import base64
                text_part = out[:m_b64.start()].strip()
                raw = m_b64.group(1)
                cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
                try:
                    img = base64.b64decode(cleaned, validate=True)
                except (binascii.Error, Exception):
                    img = None

                if img and img.startswith(b"\x89PNG\r\n\x1a\n"):
                    pid = str(uuid4())
                    self.plot_cache[pid] = img
                    return {"text": text_part, "plot_id": pid}

                return {"text": text_part}

            return {"text": out}

        except Exception as e:
            return {"text": f"Error processing query: {str(e)}", "error": str(e)}

    def get_plot(self, plot_id: str) -> Optional[bytes]:
        """Get plot data by ID."""
        return self.plot_cache.get(plot_id)

    def get_map_data(self) -> List[Dict]:
        """Get store locations for map visualization."""
        try:
            sql = """
            SELECT store_id, store_name, location, latitude AS lat, longitude AS lon
            FROM public."stores"
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY store_id
            """
            df = _run_sql_readonly(_ensure_limit(sql, 10000))
            out = []
            for _, r in df.iterrows():
                out.append({
                    "Store": r.get("store_name"),
                    "StoreId": int(r.get("store_id")) if pd.notna(r.get("store_id")) else None,
                    "Location": r.get("location"),
                    "Latitud": float(r.get("lat")),
                    "Longitud": float(r.get("lon")),
                    "Volumen": 0.5,  # Placeholder
                })
            return out
        except Exception as e:
            raise Exception(f"Error loading map data: {e}")

