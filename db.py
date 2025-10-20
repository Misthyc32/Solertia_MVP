# db.py
import os
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Date, Time, DateTime, UniqueConstraint, select, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError

# ------------------------------------------------------------
# Cómo resolvemos la URL de conexión:
# 1) SUPABASE_SP_CONN  -> Connection string del pool (recomendado)
# 2) DATABASE_URL       -> Fallback
# Nota: SUPABASE_URL y SUPABASE_KEY son para APIs HTTP de Supabase,
#       no para conectarse vía SQLAlchemy/psycopg.
# ------------------------------------------------------------
def resolve_database_url() -> str:
    url = os.getenv("SUPABASE_SP_CONN") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Falta SUPABASE_SP_CONN o DATABASE_URL. "
            "Ve a Supabase > Project Settings > Database > Connection string (psycopg)."
        )
    # Asegura SSL si no viene en la URL
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url

DATABASE_URL = resolve_database_url()

# ------------------------------------------------------------
# Engine con session pooler (sincrónico)
# ------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
Base = declarative_base()

# ------------------------------------------------------------
# MODELOS (alineados con Supabase/Postgres)
# users: ya lo tienes creado en Supabase (id, thread_id, phone, name, created_at)
# messages / reservations: tipos fuertes (date/time/timestamptz)
# ------------------------------------------------------------

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, unique=True, index=True)  # lógico por cliente
    phone = Column(String, nullable=True, index=True)
    name = Column(String, nullable=True)
    # timestamptz en DB -> usa func.now() para que lo ponga Postgres
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)   # "user" | "assistant" (puedes migrar a Enum luego)
    content = Column(Text)
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)

class Reservation(Base):
    __tablename__ = "reservations"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String, index=True, nullable=False)

    name = Column(String)      # nombre visible en la reserva
    date = Column(Date)        # YYYY-MM-DD (tipo date)
    time_start = Column(Time)  # HH:MM (tipo time)
    time_end = Column(Time)    # HH:MM (tipo time)

    # En tu tabla quedaron como start_iso/end_iso (timestamptz)
    start_iso = Column(DateTime(timezone=True), index=True)
    end_iso = Column(DateTime(timezone=True), index=True)

    people = Column(Integer)
    calendar_id = Column(String, index=True)
    event_id = Column(String, index=True)   # clave para modificar/cancelar

    status = Column(String, default="confirmed")  # confirmed|pending|canceled|failed

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("calendar_id", "event_id", name="uq_calendar_event"),
    )

# ------------------------------------------------------------
# INIT (solo crea tablas que no existan; en Supabase ya creaste users/messages/reservations)
# Puedes dejarlo por si corres localmente con otra DB.
# ------------------------------------------------------------
def init_db():
    Base.metadata.create_all(bind=engine)

# ------------------------------------------------------------
# Repos sencillos
# ------------------------------------------------------------

def save_message(session, thread_id: str, role: str, content: str):
    upsert_user(session, thread_id=thread_id)  # <-- garantiza users.row
    session.add(Message(thread_id=thread_id, role=role, content=content))
    session.commit()

def load_history(session, thread_id: str, limit: Optional[int] = None):
    q = session.execute(
        select(Message.role, Message.content)
        .where(Message.thread_id == thread_id)
        .order_by(Message.ts)
    )
    rows = q.all()
    if limit is not None:
        rows = rows[-limit:]
    return [{"role": r, "content": c} for (r, c) in rows]

def create_reservation(
    session,
    *,
    thread_id: str,
    name: Optional[str],
    date: Optional[datetime.date],
    time_start: Optional[datetime.time],
    time_end: Optional[datetime.time],
    start_iso: Optional[datetime],
    end_iso: Optional[datetime],
    people: Optional[int],
    calendar_id: Optional[str],
    event_id: Optional[str] = None,
    status: str = "confirmed",
):
    r = Reservation(
        thread_id=thread_id,
        name=name,
        date=date,
        time_start=time_start,
        time_end=time_end,
        start_iso=start_iso,
        end_iso=end_iso,
        people=people,
        calendar_id=calendar_id,
        event_id=event_id,
        status=status,
    )
    session.add(r)
    session.commit()
    return r

def set_reservation_event_id(session, reservation_id: int, event_id: str, status: str = "confirmed"):
    r = session.get(Reservation, reservation_id)
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
            .order_by(desc(Reservation.created_at))
            .limit(limit)
            .all())

def get_last_event_id_by_thread(
    session,
    thread_id: str,
    require_confirmed: bool = True
) -> Optional[str]:
    base_q = session.query(Reservation.event_id).filter(
        Reservation.thread_id == thread_id,
        Reservation.event_id.isnot(None)
    )

    if require_confirmed:
        q = base_q.filter(Reservation.status == "confirmed")
        rec = q.order_by(desc(Reservation.updated_at), desc(Reservation.created_at)).first()
        if rec and rec[0]:
            return rec[0]

    rec = base_q.order_by(desc(Reservation.updated_at), desc(Reservation.created_at)).first()
    return rec[0] if rec and rec[0] else None

def upsert_user(session, *, thread_id: str, phone: str | None = None, name: str | None = None):
    """Crea el usuario si no existe. Si existe y algunos campos están vacíos, los completa.
       No hace override de valores ya existentes (solo rellena None/'' )."""
    u = session.query(User).filter_by(thread_id=thread_id).one_or_none()
    if not u:
        u = User(thread_id=thread_id, phone=phone, name=name)
        session.add(u)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            # carrera rara: otro proceso insertó; vuelve a leer
            u = session.query(User).filter_by(thread_id=thread_id).one()
    else:
        updated = False
        if (not u.phone) and phone:
            u.phone = phone
            updated = True
        if (not u.name) and name:
            u.name = name
            updated = True
        if updated:
            session.commit()
    return u
