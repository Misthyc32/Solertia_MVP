from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from tools import CALENDAR_ID, reserva_restaurante_tool, update_reservation_tool, get_last_event_id_tool  
import datetime as dt
import json
from config import TZ
from calendar_client import get_calendar_service
from zoneinfo import ZoneInfo
from db import SessionLocal, create_reservation
# Normalizar “hoy/mañana/sábado” → YYYY-MM-DD
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
import re

WEEKDAYS = {
    "lunes": MO, "martes": TU, "miércoles": WE, "miercoles": WE,
    "jueves": TH, "viernes": FR, "sábado": SA, "sabado": SA, "domingo": SU
}
db = SessionLocal()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

class GlobalState(TypedDict):
    thread_id: str
    messages: List[dict]
    question: str
    context: List[Document]
    reservation_data: dict
    answer: str
    route: str
    pending_reservation: bool  
    pending_update: bool

reservation_prompt = ChatPromptTemplate.from_template(
    """
Eres un asistente del restaurante La Casona. Hoy es {today} ({tz}).
Hablas de manera cálida y natural, como un mesero amable.

Tu trabajo es agendar una reservación para el cliente. Necesitas estos datos:
- name (Nombre del cliente)
- date (YYYY-MM-DD o expresiones como "mañana", "hoy", "sábado")
- time_start (HH:MM en formato 24h, por ejemplo "20:00")
- time_end (HH:MM en formato 24h). Si no lo dan, asume 2 horas después.
- people (número entero de personas)
- des (Información necesaria extra para la reserva, ej. alergias, cumpleaños, celebraciones, algún dato importante)
  
Si todavía falta algún dato, haz UNA SOLA pregunta NATURAL para pedirlo (sin listas ni formatos).
Cuando ya tengas todos los datos, responde **exclusivamente** con un JSON **sin texto adicional**, exactamente con este formato:

{{
  "name": "Nombre",
  "date": "YYYY-MM-DD",
  "time_start": "HH:MM",
  "time_end": "HH:MM",
  "people": 2,
  "des": "Texto con información extra"
}}

No incluyas explicaciones, saludos ni nada fuera del JSON.
"""
)

update_prompt = ChatPromptTemplate.from_template(
    """
Eres un asistente del restaurante La Casona. Hoy es {today} ({tz}).
El cliente quiere ACTUALIZAR una reservación existente.

Datos que puedes pedir o modificar:
- name (nuevo nombre del cliente, si cambia)
- date (YYYY-MM-DD)
- time_start (HH:MM en formato 24h)
- time_end (HH:MM en formato 24h, opcional → si no, asume 2h después)
- people (número entero)
- des (Informacion necesaria extra para la reserva, ej. Alergias, Cumpleaños, Celebraciones, Algun dato que sea necesario que el restaurante sepa.)
Si falta algún dato importante para identificar o actualizar la reserva, pídeselo al cliente de forma natural (sin listas ni formatos).
Cuando ya tengas toda la información para actualizar, responde **únicamente** con un JSON limpio así:

{{
  "name": "Nuevo nombre o igual que antes",
  "date": "YYYY-MM-DD",
  "time_start": "HH:MM",
  "time_end": "HH:MM",
  "people": 2,
  "des": "Texto con información extra"
}}

No pongas explicaciones ni texto adicional fuera del JSON.
"""
)
classifier_prompt = ChatPromptTemplate.from_template(
    """You are a classifier. Classify the following message as one of two categories:
- "reservation" for booking/scheduling/mesa
- "update" for modifying/changing an existing reservation (cambiar hora, personas, etc.)
- "rag" for menu questions or recommendations.

Respond only with: reservation or update or rag.

Message: {message}"""
)

def classifier_node(state: GlobalState):
    # Si venimos en flujo de reserva pendiente, nos quedamos en reserva
    if state.get("pending_reservation") and not state.get("reservation_data", {}).get("eventId"):
        return {**state, "route": "reservation"}
    if state.get("pending_update"):
        return {**state, "route": "update"}
    
    q = state["question"]
    label = llm.invoke(classifier_prompt.format(message=q)).content.strip().lower()
    if label not in ("reservation", "rag","update"):
        label = "rag"
    return {**state, "route": label}

def classifier_router(state: GlobalState):
    if state.get("route") == "reservation":
        return "reservation_node"
    if state.get("route") == "update":
        return "update_node"
    return "retrieve"

def retrieve(state: GlobalState, vector_store):
    return {**state, "context": vector_store.similarity_search(state["question"], k=6)}

def generate(state: GlobalState):
    docs = "\n\n".join(doc.page_content for doc in state["context"])
    sys = f"Eres un asistente de restaurante llamado La Casona.\nUsa el siguiente contexto:\n{docs}"
    conversation = [{"role": "system", "content": sys}] + state["messages"] + [{"role": "user", "content": state["question"]}]
    resp = llm.invoke(conversation)
    return {**state, "answer": resp.content, "messages": state["messages"] + [{"role":"user","content":state["question"]},{"role":"assistant","content":resp.content}]}

def _combine_date_time_to_iso(date_str: str, time_str: str) -> str:
    """Convierte 'YYYY-MM-DD' + 'HH:MM' a ISO 8601 sin zona."""
    return f"{date_str}T{time_str}:00"

def _normalize_date_es(s: str, tz: str = TZ) -> str:
    s = s.strip().lower()
    today = dt.datetime.now(ZoneInfo(tz)).date()
    if s in ("hoy", "today"):
        return today.isoformat()
    if s in ("mañana", "manana", "tomorrow"):
        return (today + dt.timedelta(days=1)).isoformat()
    if s in WEEKDAYS:
        target = today + relativedelta(weekday=WEEKDAYS[s](+1))
        return target.isoformat()
    # dejar pasar YYYY-MM-DD
    try:
        dt.datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        raise ValueError("Fecha no reconocida. Usa 'hoy', 'mañana', un día (sábado), o YYYY-MM-DD.")

def _combine_date_time_to_rfc3339(date_str: str, time_str: str, tz: str = TZ) -> str:
    # Devuelve ISO con offset, ej: 2025-08-08T20:00:00-06:00
    zone = ZoneInfo(tz)
    dt_local = dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=zone)
    return dt_local.isoformat(timespec="seconds")

def _validate_reservation(appt: dict, tz: str = TZ):
    zone = ZoneInfo(tz)
    start = dt.datetime.strptime(f"{appt['date']} {appt['time_start']}", "%Y-%m-%d %H:%M").replace(tzinfo=zone)
    # usar date_end si existe
    date_end = appt.get("date_end", appt["date"])
    end   = dt.datetime.strptime(f"{date_end} {appt['time_end']}",   "%Y-%m-%d %H:%M").replace(tzinfo=zone)
    now   = dt.datetime.now(zone)
    if start < now:
        raise ValueError("La hora de inicio ya pasó. ¿Quieres otra hora?")
    if end <= start:
        raise ValueError("La hora de fin debe ser mayor a la de inicio.")
    
def _resolve_event_id_db_cache(state: GlobalState) -> Optional[str]:
    # 1) DB (confirmadas primero)
    try:
        res = get_last_event_id_tool(state["thread_id"], require_confirmed=True)  # "EVENT_ID:<id>" | "NOT_FOUND" | "ERROR:..."
        if isinstance(res, str) and res.startswith("EVENT_ID:"):
            return res.split("EVENT_ID:", 1)[1].strip()
        if res == "NOT_FOUND":
            # Fallback: permitir no-confirmadas
            res2 = get_last_event_id_tool(state["thread_id"], require_confirmed=False)
            if isinstance(res2, str) and res2.startswith("EVENT_ID:"):
                return res2.split("EVENT_ID:", 1)[1].strip()
    except Exception:
        pass

    # 2) Cache local de la conversación
    eid = state.get("reservation_data", {}).get("eventId")
    if eid:
        return eid

    return None

def _roll_messages(msgs: List[dict], limit: int = 20) -> List[dict]:
    return msgs[-limit:]

def _coerce_end_next_day_if_needed(appt: dict):
    """Si time_end <= time_start, asume cruce a día siguiente y setea date_end."""
    start = dt.datetime.strptime(f"{appt['date']} {appt['time_start']}", "%Y-%m-%d %H:%M")
    end   = dt.datetime.strptime(f"{appt['date']} {appt['time_end']}",   "%Y-%m-%d %H:%M")
    if end <= start:
        end = end + dt.timedelta(days=1)
        appt["date_end"] = end.date().isoformat()
        appt["time_end"] = end.strftime("%H:%M")
    else:
        appt["date_end"] = appt["date"]
def _extract_json_forgiving(raw: str) -> dict | None:
    """Intenta encontrar un bloque JSON { ... } dentro de texto."""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
extract_prompt = ChatPromptTemplate.from_template("""
Extrae de forma estricta un JSON con las claves:
name, date, time_start, time_end (o null), people (entero).
No agregues texto fuera del JSON.

Mensaje:
{m}
""")
def reservation_node(state: GlobalState):
    """Nodo que maneja el flujo de creación de reserva con:
    - Normalización fecha/hora
    - Manejo de cruce a medianoche
    - Validación
    - Extracción robusta de JSON
    - Persistencia en DB y llamada a Calendar
    """
    # marcar modo “pegajoso” de reserva
    if not state.get("pending_reservation"):
        state["pending_reservation"] = True

    db = SessionLocal()
    today = dt.datetime.now(ZoneInfo(TZ)).date().isoformat()

    # 1) Pedimos/normalizamos datos con el LLM (JSON-only cuando esté completo)
    convo = (
        [{"role": "system", "content": reservation_prompt.format(today=today, tz=TZ)}]
        + state["messages"]
        + [{"role": "user", "content": state["question"]}]
    )
    resp = llm.invoke(convo)
    raw = resp.content.strip()

    # 2) Intentamos obtener JSON (directo → tolerante → rescate)
    appt = None
    try:
        appt = json.loads(raw)
    except Exception:
        pass

    if appt is None:
        appt = _extract_json_forgiving(raw)

    if appt is None:
        rescue = llm.invoke(extract_prompt.format(m=raw)).content.strip()
        try:
            appt = json.loads(rescue)
        except Exception:
            appt = None

    if appt is None:
        # seguimos conversación (no hay JSON aún)
        followup = raw
        updated_messages = _roll_messages(
            state["messages"] + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": followup},
            ]
        )
        return {**state, "answer": followup, "messages": updated_messages}

    # 3) Completar/normalizar/validar
    for k in ["name", "date", "time_start", "people"]:
        if k not in appt or appt[k] in (None, "", []):
            raise ValueError(f"Falta campo obligatorio: {k}")

    if not appt.get("time_end"):
        t0 = dt.datetime.strptime(appt["time_start"], "%H:%M")
        t1 = (t0 + dt.timedelta(hours=2)).time()
        appt["time_end"] = t1.strftime("%H:%M")

    appt["date"] = _normalize_date_es(appt["date"], tz=TZ)
    _coerce_end_next_day_if_needed(appt)  # <-- maneja “00:00” o fin <= inicio
    _validate_reservation(appt, tz=TZ)

    # 4) RFC3339 con tz (usa date_end si existe)
    start_iso = _combine_date_time_to_rfc3339(appt["date"],     appt["time_start"], tz=TZ)
    end_iso   = _combine_date_time_to_rfc3339(appt.get("date_end", appt["date"]), appt["time_end"], tz=TZ)

    # 5) Tool → Calendar
    tool_result = reserva_restaurante_tool.invoke({
        "name": appt["name"],
        "start_datetime": start_iso,
        "end_datetime": end_iso,
        "num_people": int(appt["people"]),
        "des": appt.get("des", "")

    })

    # 6) Extraer event_id (si vino)
    event_id = None
    if "|EVENT_ID:" in tool_result:
        _, eid = tool_result.split("|EVENT_ID:", 1)
        event_id = eid.strip()

    # 7) Persistir en DB SIEMPRE
    status = "confirmed" if event_id else ("failed" if tool_result.startswith("❌") else "pending")
    rec = create_reservation(
        db,
        thread_id=state["thread_id"],
        name=appt["name"],
        date=appt["date"],
        time_start=appt["time_start"],
        time_end=appt["time_end"],
        start_iso=start_iso,
        end_iso=end_iso,
        people=int(appt["people"]),
        calendar_id=CALENDAR_ID,
        event_id=event_id,
        status=status
    )

    # 8) Mensaje humano
    human_msg = (
        f"¡Listo, {appt['name']}! Tu reservación para el {appt['date']}"
        f"{' (termina al día siguiente)' if appt.get('date_end') and appt['date_end'] != appt['date'] else ''} "
        f"de {appt['time_start']} a {appt['time_end']} para {appt['people']} personas. "
        f"{tool_result}"
    )

    updated_messages = _roll_messages(
        state["messages"] + [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": human_msg},
        ]
    )

    # salir de modo pegajoso si ya quedó confirmada
    if event_id:
        state["pending_reservation"] = False

    return {
        **state,
        "answer": human_msg,
        "messages": updated_messages,
        "reservation_data": {**appt, "eventId": event_id, "db_id": rec.id}
    }

def update_node(state: GlobalState):
    if not state.get("pending_update"):
        state["pending_update"] = True

    today = dt.datetime.now(ZoneInfo(TZ)).date().isoformat()

    # Resolver event_id solo desde DB o cache
    event_id = _resolve_event_id_db_cache(state)
    if not event_id:
        follow = "No encuentro tu reservación en nuestro sistema. Por favor, confirma si ya hiciste una previamente o indícame que la creemos de nuevo."
        updated_messages = _roll_messages(
            state["messages"] + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": follow},
            ]
        )
        return {**state, "answer": follow, "messages": updated_messages}

    # Preguntar/extraer JSON con cambios a aplicar
    convo = (
        [{"role": "system", "content": update_prompt.format(today=today, tz=TZ)}]
        + state["messages"]
        + [{"role": "user", "content": state["question"]}]
    )
    resp = llm.invoke(convo)
    raw = resp.content.strip()

    # Intento robusto de JSON
    upd = None
    try:
        upd = json.loads(raw)
    except Exception:
        upd = _extract_json_forgiving(raw)
        if upd is None:
            rescue = llm.invoke(extract_prompt.format(m=raw)).content.strip()
            try:
                upd = json.loads(rescue)
            except Exception:
                upd = None

    # Si aún no hay JSON, seguimos la conversación con lo que dijo el modelo
    if upd is None:
        updated_messages = _roll_messages(
            state["messages"] + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": raw},
            ]
        )
        return {**state, "answer": raw, "messages": updated_messages}

    # Normalización suave (solo campos presentes)
    norm = {}
    if upd.get("name"):
        norm["name"] = upd["name"]
    if upd.get("people") is not None:
        norm["people"] = int(upd["people"])
    if upd.get("des") is not None:
        norm["des"] = str(upd["des"])

    # Manejo de fecha/hora si están presentes
    date_str = upd.get("date")
    t0 = upd.get("time_start")
    t1 = upd.get("time_end")

    start_iso = None
    end_iso = None

    if date_str:
        date_str = _normalize_date_es(date_str, tz=TZ)

    if t0 and not t1:
        base = dt.datetime.strptime(t0, "%H:%M")
        t1 = (base + dt.timedelta(hours=2)).strftime("%H:%M")

    if date_str and t0:
        appt_tmp = {
            "date": date_str,
            "time_start": t0,
            "time_end": t1 or "00:00",
        }
        _coerce_end_next_day_if_needed(appt_tmp)
        _validate_reservation(appt_tmp, tz=TZ)

        start_iso = _combine_date_time_to_rfc3339(appt_tmp["date"], t0, tz=TZ)
        end_iso   = _combine_date_time_to_rfc3339(appt_tmp.get("date_end", appt_tmp["date"]), appt_tmp["time_end"], tz=TZ)

    # Llamada a la tool de actualización
    tool_res = update_reservation_tool.invoke({
        "event_id": event_id,
        "name": norm.get("name"),
        "num_people": norm.get("people"),
        "des": norm.get("des"),
        "start_datetime": start_iso,
        "end_datetime": end_iso,
        "tz": TZ
    })

    # Si la tool devolvió un nuevo EVENT_ID, úsalo; si no, conserva el actual
    new_event_id = event_id
    if isinstance(tool_res, str) and "|EVENT_ID:" in tool_res:
        try:
            _, eid = tool_res.split("|EVENT_ID:", 1)
            if eid.strip():
                new_event_id = eid.strip()
        except Exception:
            pass

    # Mensaje humano
    human_msg = f"¡Listo! Tu reservación ha sido actualizada. {tool_res}"

    updated_messages = _roll_messages(
        state["messages"] + [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": human_msg},
        ]
    )

    # Salimos de modo update
    state["pending_update"] = False

    # Actualiza cache local si cambió algo
    new_reservation_cache = dict(state.get("reservation_data", {}))
    if new_event_id:
        new_reservation_cache["eventId"] = new_event_id
    if date_str and t0:
        new_reservation_cache.update({
            "date": date_str,
            "time_start": t0,
            "time_end": appt_tmp["time_end"],
        })
        if appt_tmp.get("date_end"):
            new_reservation_cache["date_end"] = appt_tmp["date_end"]

    return {
        **state,
        "answer": human_msg,
        "messages": updated_messages,
        "reservation_data": new_reservation_cache
    }
# def build_app(vector_store):
#     g = StateGraph(GlobalState)
#     g.set_entry_point("classifier")
#     g.add_node("classifier", classifier_node)
#     g.add_node("retrieve", lambda s: retrieve(s, vector_store))
#     g.add_node("generate", generate)
#     g.add_node("reservation_node", reservation_node)
#     g.add_node("output", lambda s: {**s})
#     g.add_conditional_edges("classifier", classifier_router)
#     g.add_edge("retrieve", "generate")
#     g.add_edge("generate", "output")
#     g.add_edge("reservation_node", "output")
#     g.set_finish_point("output")
#     return g.compile(checkpointer=MemorySaver())
def build_app(vector_store):
    g = StateGraph(GlobalState)
    g.set_entry_point("classifier")

    g.add_node("classifier", classifier_node)
    g.add_node("retrieve", lambda s: retrieve(s, vector_store))
    g.add_node("generate", generate)
    g.add_node("reservation_node", reservation_node)
    g.add_node("update_node", update_node)       # <--- NUEVO
    g.add_node("output", lambda s: {**s})

    g.add_conditional_edges("classifier", classifier_router)

    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "output")
    g.add_edge("reservation_node", "output")
    g.add_edge("update_node", "output")          # <--- NUEVO

    g.set_finish_point("output")
    return g.compile(checkpointer=MemorySaver())
