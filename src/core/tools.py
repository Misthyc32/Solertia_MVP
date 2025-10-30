from pydantic import BaseModel, Field, validator
from langchain.tools import tool
from src.core.calendar_client import get_calendar_service, create_event, update_event, generate_calendar_invitation_link
from src.core.config import TZ
import datetime as dt
from typing import Optional
from src.core.db import SessionLocal, get_last_event_id_by_customer_id

# ID del calendario (puedes cambiarlo por el tuyo)
CALENDAR_ID = "solertia.grp@gmail.com"

class ReservaInput(BaseModel):
    first_name: str = Field(..., description="Nombre del cliente")
    start_datetime: str = Field(..., description="Inicio ISO, e.g. 2025-08-08T20:00:00-06:00")
    end_datetime: str = Field(..., description="Fin ISO, e.g. 2025-08-08T21:30:00-06:00")
    party_size: int = Field(..., ge=1, le=30)
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
    first_name: Optional[str] = Field(None, description="Nombre del cliente")
    party_size: Optional[int] = Field(None, description="Número de personas")
    des: Optional[str] = Field(None, description="Texto adicional para la descripción")
    start_datetime: Optional[str] = Field(None, description="Nuevo inicio ISO8601, ej. '2025-08-14T16:00:00-06:00'")
    end_datetime: Optional[str] = Field(None, description="Nuevo fin ISO8601, ej. '2025-08-14T18:00:00-06:00'")
    tz: Optional[str] = Field(None, description="Timezone IANA, ej. 'America/Monterrey'")

@tool("reserva_restaurante", args_schema=ReservaInput, return_direct=True)
def reserva_restaurante_tool(first_name: str, start_datetime: str, end_datetime: str, party_size: int, des:str) -> str:
    """Crea una reservación en el calendario del restaurante y devuelve un link + event_id + invitation links."""
    service = get_calendar_service()
    summary = f"Reservación: {first_name} ({party_size} personas)"
    description = f"Reservación para {party_size} personas. Creada por el agente. {des}"
    try:
        # Create event in restaurant calendar
        ev = create_event(service, CALENDAR_ID, summary, description, start_datetime, end_datetime, tz=TZ)
        restaurant_link = ev.get("htmlLink", "")
        eid = ev.get("id", "")
        
        # Generate invitation links for the user
        invitation_links = generate_calendar_invitation_link(summary, start_datetime, end_datetime, first_name, party_size, des, tz=TZ)
        
        # Format the response with both restaurant link and invitation links
        response = f"✅ Reservación creada exitosamente!\n\n"
        response += f"📅 **Vista en nuestro calendario:** {restaurant_link}\n\n"
        response += f"📱 **Agregar a tu calendario:**\n"
        response += f"• Google Calendar: {invitation_links['google']}\n"
        response += f"• Outlook/Hotmail: {invitation_links['outlook']}\n"
        response += f"• Apple Calendar: {invitation_links['apple']}\n\n"
        response += f"Event ID: {eid}"
        
        return f"{response}|EVENT_ID:{eid}"
    except Exception as e:
        return f"❌ Error al crear la reservación: {e}"

@tool("update_reserva_restaurante", args_schema=ReservaUpdateInput, return_direct=True)
def update_reservation_tool(event_id: str,first_name: Optional[str] = None,party_size: Optional[int] = None,des: Optional[str] = None,start_datetime: Optional[str] = None,end_datetime: Optional[str] = None,tz: Optional[str] = None
) -> str:
    """
    Actualiza una reservación en el calendario del restaurante y devuelve un link + event_id.
    Solo los campos provistos serán aplicados; los no provistos se conservan.
    """
    try:
        service = get_calendar_service()

        # Construye summary solo si hay datos nuevos
        new_summary = None
        if first_name is not None or party_size is not None:
            # Si alguno falta, lo omitimos y dejamos que update_event conserve el original
            # (por eso no intentamos leer el evento aquí; dejamos a update_event mantener defaults)
            # Pero si quieres forzar un summary con lo disponible:
            if first_name is not None and party_size is not None:
                new_summary = f"Reservación: {first_name} ({party_size} personas)"
            elif first_name is not None:
                new_summary = f"Reservación: {first_name}"
            elif party_size is not None:
                new_summary = f"Reservación: ({party_size} personas)"

        # Construye description solo si hay algo que añadir/cambiar
        new_description = None
        if des is not None or party_size is not None:
            # Texto base, añade solo lo que venga
            parts = []
            if party_size is not None:
                parts.append(f"Reservación para {party_size} personas.")
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

        restaurant_link = ev.get("htmlLink", "")
        eid = ev.get("id", event_id)
        
        # Generate new invitation links for the updated event
        updated_summary = ev.get("summary", f"Reservación actualizada")
        updated_start = ev.get("start", {}).get("dateTime", start_datetime)
        updated_end = ev.get("end", {}).get("dateTime", end_datetime)
        
        # Extract name and people from the updated summary or use defaults
        updated_name = first_name if first_name else "Cliente"
        updated_people = party_size if party_size else 2
        updated_des = des if des else ""
        
        invitation_links = generate_calendar_invitation_link(
            updated_summary, updated_start, updated_end, updated_name, updated_people, updated_des, tz_effective
        )
        
        # Format the response with both restaurant link and invitation links
        response = f"✅ Reservación actualizada exitosamente!\n\n"
        response += f"📅 **Vista en nuestro calendario:** {restaurant_link}\n\n"
        response += f"📱 **Agregar a tu calendario actualizado:**\n"
        response += f"• Google Calendar: {invitation_links['google']}\n"
        response += f"• Outlook/Hotmail: {invitation_links['outlook']}\n"
        response += f"• Apple Calendar: {invitation_links['apple']}\n\n"
        response += f"Event ID: {eid}"

        return f"{response}|EVENT_ID:{eid}"

    except Exception as e:
        return f"Error al actualizar la reservación: {e}"
    
def get_last_event_id_tool(thread_id: str, require_confirmed: bool = True) -> str:
    """
    Busca en la DB el EVENT_ID más reciente para este thread_id (customer_id).
    Devuelve:
      - "EVENT_ID:<id>" si existe
      - "NOT_FOUND" si no hay
      - "ERROR:<msg>" ante excepción
    
    Args:
        thread_id: Customer/Thread ID (unique identifier for the customer)
        require_confirmed: Si True, solo busca reservas confirmadas
    """
    session = SessionLocal()
    try:
        eid = get_last_event_id_by_customer_id(session, thread_id, require_confirmed=require_confirmed)
        return f"EVENT_ID:{eid}" if eid else "NOT_FOUND"
    except Exception as e:
        return f"ERROR:{e}"
    finally:
        session.close()