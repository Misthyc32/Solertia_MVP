from pydantic import BaseModel, Field, validator
from langchain.tools import tool
from calendar_client import get_calendar_service, create_event, update_event
from config import TZ
import datetime as dt
import json
from typing import Optional
from db import SessionLocal, get_last_event_id_by_thread

# ID del calendario (puedes cambiarlo por el tuyo)
CALENDAR_ID = "solertia.grp@gmail.com"

class ReservaInput(BaseModel):
    name: str = Field(..., description="Nombre del cliente")
    start_datetime: str = Field(..., description="Inicio ISO, e.g. 2025-08-08T20:00:00-06:00")
    end_datetime: str = Field(..., description="Fin ISO, e.g. 2025-08-08T21:30:00-06:00")
    num_people: int = Field(..., ge=1, le=30)
    des: str = Field(..., description="Consideraciones para guardar en la reserva")
    @validator("start_datetime", "end_datetime")
    def _iso_ok(cls, v):
        try:
            dt.datetime.fromisoformat(v.replace("Z","+00:00"))
        except Exception:
            raise ValueError("Debe ser ISO 8601 (YYYY-MM-DDTHH:MM:SS±HH:MM).")
        return v
class ReservaUpdateInput(BaseModel):
    event_id: str = Field(..., description="ID del evento a actualizar en Google Calendar")
    name: Optional[str] = Field(None, description="Nombre del cliente")
    num_people: Optional[int] = Field(None, description="Número de personas")
    des: Optional[str] = Field(None, description="Texto adicional para la descripción")
    start_datetime: Optional[str] = Field(None, description="Nuevo inicio ISO8601, ej. '2025-08-14T16:00:00-06:00'")
    end_datetime: Optional[str] = Field(None, description="Nuevo fin ISO8601, ej. '2025-08-14T18:00:00-06:00'")
    tz: Optional[str] = Field(None, description="Timezone IANA, ej. 'America/Monterrey'")

@tool("reserva_restaurante", args_schema=ReservaInput, return_direct=True)
def reserva_restaurante_tool(name: str, start_datetime: str, end_datetime: str, num_people: int, des:str) -> str:
    """Crea una reservación en el calendario del restaurante y devuelve un link + event_id."""
    service = get_calendar_service()
    summary = f"Reservación: {name} ({num_people} personas)"
    description = f"Reservación para {num_people} personas. Creada por el agente. {des}"
    try:
        ev = create_event(service, CALENDAR_ID, summary, description, start_datetime, end_datetime, tz=TZ)
        link = ev.get("htmlLink", "")
        eid  = ev.get("id", "")
        return f"Reservación creada: {link}|EVENT_ID:{eid}"
    except Exception as e:
        return f"Error al crear la reservación: {e}"

@tool("update_reserva_restaurante", args_schema=ReservaUpdateInput, return_direct=True)
def update_reservation_tool(event_id: str,name: Optional[str] = None,num_people: Optional[int] = None,des: Optional[str] = None,start_datetime: Optional[str] = None,end_datetime: Optional[str] = None,tz: Optional[str] = None
) -> str:
    """
    Actualiza una reservación en el calendario del restaurante y devuelve un link + event_id.
    Solo los campos provistos serán aplicados; los no provistos se conservan.
    """
    try:
        service = get_calendar_service()

        # Construye summary solo si hay datos nuevos
        new_summary = None
        if name is not None or num_people is not None:
            # Si alguno falta, lo omitimos y dejamos que update_event conserve el original
            # (por eso no intentamos leer el evento aquí; dejamos a update_event mantener defaults)
            # Pero si quieres forzar un summary con lo disponible:
            if name is not None and num_people is not None:
                new_summary = f"Reservación: {name} ({num_people} personas)"
            elif name is not None:
                new_summary = f"Reservación: {name}"
            elif num_people is not None:
                new_summary = f"Reservación: ({num_people} personas)"

        # Construye description solo si hay algo que añadir/cambiar
        new_description = None
        if des is not None or num_people is not None:
            # Texto base, añade solo lo que venga
            parts = []
            if num_people is not None:
                parts.append(f"Reservación para {num_people} personas.")
            if des is not None:
                parts.append(des)
            # Marca que viene del agente
            parts.append("Actualizada por el agente.")
            new_description = " ".join(parts).strip()

        tz_effective = tz if tz is not None else TZ

        ev = update_event(
            service=service,
            calendar_id=CALENDAR_ID,
            event_id=event_id,
            summary=new_summary,
            description=new_description,
            start_iso=start_datetime,
            end_iso=end_datetime,
            tz=tz_effective,
        )

        link = ev.get("htmlLink", "")
        eid  = ev.get("id", event_id)

        return f"Reservación actualizada: {link}|EVENT_ID:{eid}"

    except Exception as e:
        return f"Error al actualizar la reservación: {e}"
    
def get_last_event_id_tool(thread_id: str, require_confirmed: bool = True) -> str:
    """
    Busca en la DB el EVENT_ID más reciente para este thread_id.
    Devuelve:
      - "EVENT_ID:<id>" si existe
      - "NOT_FOUND" si no hay
      - "ERROR:<msg>" ante excepción
    """
    session = SessionLocal()
    try:
        eid = get_last_event_id_by_thread(session, thread_id, require_confirmed=require_confirmed)
        return f"EVENT_ID:{eid}" if eid else "NOT_FOUND"
    except Exception as e:
        return f"ERROR:{e}"
    finally:
        session.close()