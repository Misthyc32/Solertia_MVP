from db import init_db, SessionLocal, list_reservations_by_thread
from graph import _normalize_date_es, _combine_date_time_to_rfc3339, _validate_reservation
from config import TZ

# Inicializar DB
init_db()
db = SessionLocal()

# Simular datos del usuario
appt = {
    "name": "Eduardo",
    "date": "mañana",
    "time_start": "20:00",
    "time_end": "22:00",
    "people": 4
}

# Normalizar y validar
appt["date"] = _normalize_date_es(appt["date"], tz=TZ)
_validate_reservation(appt, tz=TZ)

# Combinar a RFC3339
start_iso = _combine_date_time_to_rfc3339(appt["date"], appt["time_start"], tz=TZ)
end_iso   = _combine_date_time_to_rfc3339(appt["date"], appt["time_end"], tz=TZ)

print("START ISO:", start_iso)
print("END ISO:", end_iso)

# Guardar en DB manualmente (sin pasar por Calendar)
from db import create_reservation
rec = create_reservation(
    db,
    thread_id="user-123",
    name=appt["name"],
    date=appt["date"],
    time_start=appt["time_start"],
    time_end=appt["time_end"],
    start_iso=start_iso,
    end_iso=end_iso,
    people=appt["people"],
    calendar_id="test_calendar",
    event_id="fake_event_001",
    status="confirmed"
)
print("Reservation ID in DB:", rec.id)

# Listar reservas de este usuario
rows = list_reservations_by_thread(db, "user-123")
for r in rows:
    print(r.id, r.name, r.date, r.start_iso, r.event_id)

#Probar nodo reservation
from graph import reservation_node

state = {
    "thread_id": "user-123",
    "messages": [],
    "question": "Quiero una mesa para mañana de las 8pm a las 10pm para 3 personas a nombre de Ana",
    "reservation_data": {},
    "answer": ""
}

new_state = reservation_node(state)
print("ANSWER:", new_state["answer"])
print("DB ID:", new_state["reservation_data"].get("db_id"))
print("EVENT ID:", new_state["reservation_data"].get("eventId"))
