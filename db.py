# db.py
import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text,
    UniqueConstraint, select, desc
)
from typing import Optional
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sales_ai.db")

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, unique=True, index=True)  # tu ID lógico por cliente
    phone = Column(String, nullable=True, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, index=True)
    role = Column(String)  # "user" | "assistant"
    content = Column(Text)
    ts = Column(DateTime, default=datetime.utcnow, index=True)

class Reservation(Base):
    __tablename__ = "reservations"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, index=True)
    name = Column(String)               # nombre que aparece en la reserva
    date = Column(String)               # "YYYY-MM-DD"
    time_start = Column(String)         # "HH:MM"
    time_end = Column(String)           # "HH:MM"
    start_iso = Column(String)          # RFC3339 con tz
    end_iso = Column(String)            # RFC3339 con tz
    people = Column(Integer)
    calendar_id = Column(String)        # tu CALENDAR_ID
    event_id = Column(String, index=True)  # <-- clave para modificar/cancelar
    status = Column(String, default="confirmed")  # confirmed | pending | canceled | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("calendar_id", "event_id", name="uq_calendar_event"),
    )

def init_db():
    Base.metadata.create_all(bind=engine)

# Helpers “repo” sencillos
def ensure_user(session, thread_id: str, phone: str | None = None, name: str | None = None):
    u = session.query(User).filter_by(thread_id=thread_id).one_or_none()
    if not u:
        u = User(thread_id=thread_id, phone=phone, name=name)
        session.add(u)
        session.commit()
    return u

def save_message(session, thread_id: str, role: str, content: str):
    session.add(Message(thread_id=thread_id, role=role, content=content))
    session.commit()

def load_history(session, thread_id:str):
    contents = session.execute(
    select(Message.content).where(Message.thread_id == thread_id).order_by(Message.ts)
    ).scalars().all()
    roles = session.execute(
        select(Message.role).where(Message.thread_id == thread_id).order_by(Message.ts)
    ).scalars().all()
    history=[]

    for role,content in zip(roles,contents):
        history.append({"role": role, "content": content})
        
    return history

def create_reservation(session, *, thread_id, name, date, time_start, time_end,
                       start_iso, end_iso, people, calendar_id, event_id=None, status="confirmed"):
    r = Reservation(
        thread_id=thread_id, name=name, date=date, time_start=time_start, time_end=time_end,
        start_iso=start_iso, end_iso=end_iso, people=people, calendar_id=calendar_id,
        event_id=event_id, status=status
    )
    session.add(r)
    session.commit()
    return r

def set_reservation_event_id(session, reservation_id: int, event_id: str, status: str = "confirmed"):
    r = session.query(Reservation).get(reservation_id)
    if r:
        r.event_id = event_id
        r.status = status
        session.commit()
    return r

def find_reservation_by_event_id(session, event_id: str):
    return session.query(Reservation).filter_by(event_id=event_id).one_or_none()

def list_reservations_by_thread(session, thread_id: str, limit: int = 50):
    return (session.query(Reservation)
            .filter_by(thread_id=thread_id)
            .order_by(Reservation.created_at.desc())
            .limit(limit)
            .all())

def get_last_event_id_by_thread(
    session,
    thread_id: str,
    require_confirmed: bool = True
) -> Optional[str]:
    """Devuelve el event_id más reciente (por updated_at/created_at) para el thread dado.
       Prioriza reservas con status='confirmed' y event_id no-nulo.
       Si no hay confirmadas y require_confirmed=False, toma la más reciente con event_id."""
    base_q = session.query(Reservation.event_id).filter(
        Reservation.thread_id == thread_id,
        Reservation.event_id.isnot(None)
    )

    if require_confirmed:
        q = base_q.filter(Reservation.status == "confirmed")
        rec = q.order_by(desc(Reservation.updated_at), desc(Reservation.created_at)).first()
        if rec and rec[0]:
            return rec[0]

    # Fallback: cualquiera con event_id
    rec = base_q.order_by(desc(Reservation.updated_at), desc(Reservation.created_at)).first()
    return rec[0] if rec and rec[0] else None