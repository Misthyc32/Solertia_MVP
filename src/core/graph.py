from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from src.core.tools import CALENDAR_ID, reserva_restaurante_tool, update_reservation_tool, get_last_event_id_tool  
import datetime as dt
import json
from src.core.config import TZ
from zoneinfo import ZoneInfo
from src.core.db import SessionLocal, create_reservation, upsert_user
# Normalizar “hoy/mañana/sábado” → YYYY-MM-DD
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
import re

WEEKDAYS = {
    "lunes": MO, "martes": TU, "miércoles": WE, "miercoles": WE,
    "jueves": TH, "viernes": FR, "sábado": SA, "sabado": SA, "domingo": SU
}

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

class GlobalState(TypedDict):
    customer_id: str
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
- date (YYYY-MM-DD o expresiones como "mañana", "pasado mañana", "hoy", o día de la semana como "viernes", "sábado" que significa el próximo ese día)
- time_start (HH:MM en formato 24h, por ejemplo "20:00")
- time_end (HH:MM en formato 24h). Si no lo dan, asume 2 horas después.
- people (número entero de personas)
- des (Información necesaria extra para la reserva, ej. alergias, cumpleaños, celebraciones, algún dato importante)

IMPORTANTE sobre fechas: Si el cliente dice "viernes", "sábado", etc., significa el PRÓXIMO día de esa semana que sea hoy o después. Si hoy es lunes y dicen "viernes", significa el viernes de esta semana (no el de la próxima semana).
  
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
    """You are a classifier for a restaurant assistant. Classify the following message as one of three categories:

- "reservation" for booking/scheduling/reserving a table (reservar, mesa, cita, agendar, reservación, reserva, quiero ir, necesito mesa, etc.)
- "update" for modifying/changing an existing reservation (cambiar hora, personas, modificar, actualizar, etc.)
- "rag" for menu questions, recommendations, or general restaurant information

Examples:
- "Quiero hacer una reservación" → reservation
- "Reservar mesa para 4 personas" → reservation
- "Necesito una mesa" → reservation
- "¿Qué platillos tienen?" → rag
- "Recomiéndame algo" → rag
- "Cambiar mi reserva" → update

Respond only with: reservation or update or rag.

Message: {message}"""
)

def classifier_node(state: GlobalState):
    # Si venimos en flujo de reserva pendiente, SIEMPRE nos quedamos en reserva
    # hasta que la reserva sea confirmada (tenga eventId)
    if state.get("pending_reservation") and not state.get("reservation_data", {}).get("eventId"):
        print(f"🔒 Maintaining reservation mode - pending_reservation=True, eventId missing")
        return {**state, "route": "reservation"}
    if state.get("pending_update"):
        return {**state, "route": "update"}
    
    q = state["question"].lower()
    
    # Si es un nuevo cliente sin historial, clasificar como RAG a menos que tenga palabras clave de reserva
    if not state.get("messages") or len(state.get("messages", [])) == 0:
        print(f"🆕 New conversation, checking if reservation keywords present")
    
    # Palabras clave para detectar reservaciones (expandidas)
    reservation_keywords = [
        "reservar", "reservación", "reserva", "mesa", "cita", "agendar", 
        "quiero ir", "necesito mesa", "para personas", "hoy", "mañana",
        "cumpleaños", "celebrar", "cenar", "comer", "viernes", "sábado",
        "lunes", "martes", "miércoles", "jueves", "domingo", "fecha", "hora",
        "tengo", "personas", "pm", "am", "el día", "el", "de"
    ]
    
    # Palabras clave para detectar actualizaciones
    update_keywords = [
        "cambiar", "modificar", "actualizar", "cambio", "diferente"
    ]
    
    # Verificar si contiene palabras clave de reservación
    if any(keyword in q for keyword in reservation_keywords):
        print(f"🔵 Classified as RESERVATION based on keywords")
        return {**state, "route": "reservation"}
    
    # Verificar si contiene palabras clave de actualización
    if any(keyword in q for keyword in update_keywords):
        print(f"🔵 Classified as UPDATE based on keywords")
        return {**state, "route": "update"}
    
    # Usar el LLM como fallback, pero buscar más palabras
    # Si el mensaje contiene números o fechas, probablemente es una reserva
    import re
    has_time = bool(re.search(r'\d+:\d+|\d+\s*(pm|am|PM|AM)', q))
    has_number = bool(re.search(r'\d+', q))
    
    if has_time or (has_number and len(q) > 5):
        print(f"🔵 Contains time/number, classifying as RESERVATION")
        return {**state, "route": "reservation"}
    
    try:
        label = llm.invoke(classifier_prompt.format(message=state["question"])).content.strip().lower()
        if label not in ("reservation", "rag", "update"):
            label = "rag"
        print(f"🔵 LLM classified as: {label}")
    except Exception:
        label = "rag"
        print(f"🔵 Exception in LLM classification, defaulting to: {label}")
    
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
        res = get_last_event_id_tool(state["customer_id"], require_confirmed=True)  # "EVENT_ID:<id>" | "NOT_FOUND" | "ERROR:..."
        if isinstance(res, str) and res.startswith("EVENT_ID:"):
            return res.split("EVENT_ID:", 1)[1].strip()
        if res == "NOT_FOUND":
            # Fallback: permitir no-confirmadas
            res2 = get_last_event_id_tool(state["customer_id"], require_confirmed=False)
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
    - Persistencia en DB (Supabase) y llamada a Calendar
    """
    # 0) Marcar modo “pegajoso” de reserva
    if not state.get("pending_reservation"):
        state["pending_reservation"] = True

    db = SessionLocal()
    try:
        today = dt.datetime.now(ZoneInfo(TZ)).date().isoformat()

        # 1) Pedimos/normalizamos datos con el LLM (JSON-only cuando esté completo)
        convo = (
            [{"role": "system", "content": reservation_prompt.format(today=today, tz=TZ)}]
            + state["messages"]
            + [{"role": "user", "content": state["question"]}]
        )
        resp = llm.invoke(convo)
        raw = (resp.content or "").strip()

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
            return {**state, "answer": followup, "messages": updated_messages, "pending_reservation": True}

        # 3) Completar/normalizar/validar
        try:
            missing_fields = []
            for k in ["name", "date", "time_start", "people"]:
                if k not in appt or appt[k] in (None, "", []):
                    missing_fields.append(k)
            
            if missing_fields:
                raise ValueError(f"Faltan campos obligatorios: {', '.join(missing_fields)}")
        except ValueError as e:
            # Si falta información, pedirla pero mantener en modo reserva
            error_msg = f"Necesito más información para completar tu reserva: {str(e)}"
            updated_messages = _roll_messages(
                state["messages"] + [
                    {"role": "user", "content": state["question"]},
                    {"role": "assistant", "content": error_msg},
                ]
            )
            return {**state, "answer": error_msg, "messages": updated_messages, "pending_reservation": True}

        # Si falta hora de fin, colocar 2h por defecto
        if not appt.get("time_end"):
            t0 = dt.datetime.strptime(appt["time_start"], "%H:%M")
            t1 = (t0 + dt.timedelta(hours=2)).time()
            appt["time_end"] = t1.strftime("%H:%M")

        # Normalizar fecha en español y coherencia de fin al día siguiente si corresponde
        appt["date"] = _normalize_date_es(appt["date"], tz=TZ)
        _coerce_end_next_day_if_needed(appt)  # maneja “00:00” o fin <= inicio
        _validate_reservation(appt, tz=TZ)

        # 4) RFC3339 con tz (usa date_end si existe)
        start_iso_str = _combine_date_time_to_rfc3339(appt["date"],     appt["time_start"], tz=TZ)
        end_iso_str   = _combine_date_time_to_rfc3339(appt.get("date_end", appt["date"]), appt["time_end"], tz=TZ)

        # Convertir a datetime aware (por si la DB usa timestamptz)
        # Maneja "Z" -> "+00:00" si fuera necesario
        start_iso_str = start_iso_str.replace("Z", "+00:00")
        end_iso_str   = end_iso_str.replace("Z", "+00:00")
        start_dt = dt.datetime.fromisoformat(start_iso_str)
        end_dt   = dt.datetime.fromisoformat(end_iso_str)

        # 5) Tool → Calendar
        tool_result = reserva_restaurante_tool.invoke({
            "first_name": appt["name"],
            "start_datetime": start_iso_str,
            "end_datetime": end_iso_str,
            "party_size": int(appt["people"]),
            "des": appt.get("des", "")
        })

        # 6) Extraer event_id (si vino)
        event_id = None
        if "|EVENT_ID:" in tool_result:
            _, eid = tool_result.split("|EVENT_ID:", 1)
            event_id = eid.strip()

        # 7) Asegurar usuario y Persistir en DB SIEMPRE
        #    Usa datos del estado si existen; de lo contrario, rellena con lo que viene del appt
        print(f"🔍 Attempting user upsert for customer_id={state.get('customer_id')}")
        
        try:
            customer_id_int = int(state["customer_id"])
            
            # Extract first_name and last_name from the name
            full_name = state.get("user_name") or appt.get("name", "")
            name_parts = full_name.split(maxsplit=1) if full_name else []
            first_name = name_parts[0] if len(name_parts) > 0 else full_name
            last_name = name_parts[1] if len(name_parts) > 1 else None
            
            print(f"📝 Processing reservation for customer_id={customer_id_int}, name='{full_name}' -> first='{first_name}', last='{last_name}'")
            print(f"📞 Phone from state: {state.get('phone')}, WhatsApp: {state.get('whatsapp')}")
            
            upsert_user(
                db,
                customer_id=customer_id_int,
                whatsapp=state.get("phone") or state.get("whatsapp"),
                first_name=first_name,
                last_name=last_name,
            )
        except (ValueError, TypeError) as e:
            print(f"⚠️  Could not process customer_id={state.get('customer_id')}: {e}")
        except Exception as e:
            print(f"❌ Error upserting user: {e}")
            import traceback
            traceback.print_exc()

        # Parse a tipos fuertes para columnas Postgres (date, time)
        date_obj = dt.date.fromisoformat(appt["date"])        # 'YYYY-MM-DD'
        t_start = dt.datetime.strptime(appt["time_start"], "%H:%M").time()
        t_end   = dt.datetime.strptime(appt["time_end"], "%H:%M").time()

        status = "confirmed" if event_id else ("failed" if tool_result.startswith("❌") else "pending")
        
        print(f"💾 Creating reservation: customer_id={state['customer_id']}, event_id={event_id}, status={status}")
        
        rec = None
        try:
            rec = create_reservation(
                db,
                customer_id=state["customer_id"],
                name=appt["name"],
                date=date_obj,
                time_start=t_start,
                time_end=t_end,
                start_iso=start_dt,
                end_iso=end_dt,
                party_size=int(appt["people"]),
                calendar_id=CALENDAR_ID,
                event_id=event_id,
                status=status
            )
            print(f"✅ Reservation created successfully: reservation_id={rec.reservation_id}")
        except Exception as e:
            print(f"❌ Error creating reservation: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - continue with the response even if DB save fails

        # 8) Mensaje humano
        crosses_day = (appt.get("date_end") and appt["date_end"] != appt["date"])
        
        # Extract links from tool_result (remove EVENT_ID part)
        links_text = tool_result.split("|EVENT_ID:")[0].strip() if "|EVENT_ID:" in tool_result else tool_result
        
        print(f"📋 Tool result: {tool_result}")
        print(f"🔗 Links text: {links_text}")
        
        human_msg = (
            f"¡Listo, {appt['name']}! Tu reservación para el {appt['date']}"
            f"{' (termina al día siguiente)' if crosses_day else ''} "
            f"de {appt['time_start']} a {appt['time_end']} para {appt['people']} personas.\n\n"
            f"{links_text}"
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
            "reservation_data": {**appt, "eventId": event_id, "db_id": rec.reservation_id if rec else None}
        }

    finally:
        db.close()
def update_node(state: GlobalState):
    if not state.get("pending_update"):
        state["pending_update"] = True

    db = SessionLocal()
    try:
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
            "first_name": norm.get("name"),
            "party_size": norm.get("people"),
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

        # ACTUALIZAR LA BASE DE DATOS POSTGRESQL
        from src.core.db import find_reservation_by_event_id
        
        # Buscar la reservación en la DB por event_id
        reservation = find_reservation_by_event_id(db, event_id)
        
        if reservation:
            # Actualizar solo los campos que cambiaron
            updated_fields = []
            
            if norm.get("name") and norm["name"] != reservation.name:
                reservation.name = norm["name"]
                updated_fields.append("name")
                
            if norm.get("people") is not None and norm["people"] != reservation.party_size:
                reservation.party_size = norm["people"]
                updated_fields.append("party_size")
                
            if date_str and t0:
                # Actualizar fechas y horas
                new_date_obj = dt.date.fromisoformat(date_str)
                new_time_start = dt.datetime.strptime(t0, "%H:%M").time()
                new_time_end = dt.datetime.strptime(appt_tmp["time_end"], "%H:%M").time()
                
                if new_date_obj != reservation.date:
                    reservation.date = new_date_obj
                    updated_fields.append("date")
                    
                if new_time_start != reservation.time_start:
                    reservation.time_start = new_time_start
                    updated_fields.append("time_start")
                    
                if new_time_end != reservation.time_end:
                    reservation.time_end = new_time_end
                    updated_fields.append("time_end")
                
                # Actualizar timestamps ISO
                if start_iso:
                    start_dt = dt.datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                    reservation.start_iso = start_dt
                    updated_fields.append("start_iso")
                    
                if end_iso:
                    end_dt = dt.datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                    reservation.end_iso = end_dt
                    updated_fields.append("end_iso")
            
            # Actualizar el event_id si cambió
            if new_event_id != event_id:
                reservation.event_id = new_event_id
                updated_fields.append("event_id")
            
            # Actualizar timestamp de modificación
            if updated_fields:
                reservation.updated_at = dt.datetime.now(ZoneInfo(TZ))
                db.commit()
                print(f"Database updated for reservation {reservation.reservation_id}: {', '.join(updated_fields)}")
            else:
                print("No database fields needed updating")
        else:
            print(f"Warning: Could not find reservation with event_id {event_id} in database")

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
    
    finally:
        db.close()
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
