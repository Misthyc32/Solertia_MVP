# db.py
import os
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    Float, Numeric, BigInteger, create_engine, Column, Integer, String, Text,
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
        # Default to SQLite for local development
        url = "sqlite:///./solertia_local.db"
        print("⚠️  No database URL found. Using SQLite for local development.")
        print("   To use PostgreSQL/Supabase, set DATABASE_URL or SUPABASE_SP_CONN environment variable.")
    else:
        print(f"✅ Using database: {url[:20]}...")
    
    # Only add SSL mode for PostgreSQL URLs
    if url.startswith("postgresql://") and "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    
    return url

DATABASE_URL = resolve_database_url()

# ------------------------------------------------------------
# Engine con session pooler (sincrónico)
# ------------------------------------------------------------
# Configure engine based on database type
if DATABASE_URL.startswith("sqlite://"):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        future=True,
    )
else:
    # PostgreSQL configuration
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

class User(Base):
    """
    Customer/User model representing restaurant customers.
    
    Maps to the 'customers' table in Supabase/PostgreSQL.
    
    Attributes:
        customer_id (int): Primary key, unique customer identifier
        first_name (str): Customer's first name
        last_name (str): Customer's last name
        email (str): Customer's email address
        whatsapp (str): Customer's WhatsApp number for notifications
        birth_date (str): Customer's birth date (stored as text)
        ticket_promedio_cliente (Decimal): Average spending per customer
        created_at (datetime): Timestamp when customer record was created
        
    Note:
        - created_at is automatically set by the database
        - customer_id is the primary key and must be unique
    """
    __tablename__ = "customers"
    customer_id = Column(BigInteger, primary_key=True)
    first_name = Column(Text, nullable=True)  
    last_name = Column(Text, nullable=True)   
    email = Column(Text, nullable=True)       
    whatsapp = Column(Text, nullable=True)    
    birth_date = Column(Text, nullable=True)  
    ticket_promedio_cliente = Column(Numeric, nullable=True)  
    created_at = Column(DateTime(timezone=True), nullable=True, server_default=func.now())  

class Message(Base):
    """
    Message model for storing chat conversation history.
    
    Maps to the 'messages' table in Supabase/PostgreSQL.
    Stores all messages exchanged between users and the assistant.
    
    Attributes:
        id (int): Primary key, auto-incremented message ID
        customer_id (str): Foreign key to customers table, identifies the user
        role (str): Message role - 'user' for customer messages, 'assistant' for bot responses
        content (str): The actual message text content
        ts (datetime): Timestamp when message was sent (auto-generated)
        
    Indexes:
        - customer_id: Indexed for fast lookups by customer
        - ts: Indexed for sorting conversations by time
        
    Example:
        Message(customer_id="123", role="user", content="I want to make a reservation")
    """
    __tablename__ = "messages"
    id = Column(BigInteger, primary_key=True,autoincrement=True)
    customer_id = Column(Text, index=True, nullable=False)
    role = Column(Text, nullable=False)   # "user" | "assistant" (puedes migrar a Enum luego)
    content = Column(Text)
    ts = Column(DateTime(timezone=True),nullable=True, server_default=func.now(), index=True)

class Reservation(Base):
    """
    Reservation model for restaurant table bookings.
    
    Maps to the 'reservations' table in Supabase/PostgreSQL.
    Stores reservation details including customer, time, party size, and payment info.
    
    Core Attributes:
        reservation_id (int): Primary key, unique reservation identifier
        customer_id (int): Foreign key to customers table
        store_id (int): ID of the restaurant location/store
        reservation_time (str): Human-readable reservation time (e.g., "8:00 PM")
        party_size (int): Number of people in the reservation
        status (str): Reservation status ('confirmed', 'pending', 'canceled', 'completed')
        special_requests (str): Special requests or notes (allergies, celebrations, etc.)
        waiter_id (int): Assigned waiter ID
        table_num (int): Assigned table number
        tip (Decimal): Tip amount for the reservation
        total_ticket (Decimal): Total bill amount
        payment_method (str): Payment method used ('cash', 'card', 'digital', etc.)
        
    Legacy Calendar Attributes (for Google Calendar integration):
        name (str): Customer name for calendar events
        date (date): Reservation date
        time_start (time): Start time for the reservation
        time_end (time): End time for the reservation
        start_iso (datetime): ISO format start datetime with timezone
        end_iso (datetime): ISO format end datetime with timezone
        calendar_id (str): Google Calendar ID
        event_id (str): Google Calendar event ID
        
    Timestamps:
        created_at (datetime): When reservation was created (auto-generated)
        updated_at (datetime): Last update timestamp (auto-updated)
        
    Example:
        Reservation(
            customer_id=123,
            party_size=4,
            name="John Doe",
            date=date(2024, 12, 25),
            time_start=time(20, 0),
            time_end=time(22, 0),
            status="confirmed"
        )
    """
    __tablename__ = "reservations"
    reservation_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(BigInteger, nullable=True)
    store_id = Column(Integer, nullable=True)
    reservation_time = Column(Text, nullable=True)  
    party_size = Column(Integer, nullable=True)
    status = Column(Text, default="confirmed", nullable=True)  
    special_requests = Column(Text, nullable=True) 
    waiter_id = Column(Integer, nullable=True)
    table_num = Column(Integer, nullable=True)  
    tip = Column(Numeric, nullable=True)  
    total_ticket = Column(Numeric, nullable=True)  
    payment_method = Column(Text, nullable=True)  
    
    # Legacy fields
    name = Column(Text, nullable=True)  
    date = Column(Date, nullable=True)
    time_start = Column(Time, nullable=True)
    time_end = Column(Time, nullable=True)
    start_iso = Column(DateTime(timezone=True), nullable=True, index=True)
    end_iso = Column(DateTime(timezone=True), nullable=True, index=True)
    calendar_id = Column(Text, nullable=True, index=True)  
    event_id = Column(Text, nullable=True, index=True)  
    
    created_at = Column(DateTime(timezone=True), nullable=True, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=True, server_default=func.now(), onupdate=func.now())


def init_db():
    Base.metadata.create_all(bind=engine)


def save_message(session, customer_id: str, role: str, content: str):
    # Try to upsert user first (this might fail if customer_id is too large)
    try:
        customer_id_int = int(customer_id)
        upsert_user(session, customer_id=customer_id_int)
    except (ValueError, TypeError, Exception) as e:
        # If user creation fails, log it but continue with message saving
        print(f"Warning: Could not create user for customer_id={customer_id}: {e}")
    
    # Always save the message, even if user creation failed
    try:
        session.add(Message(customer_id=customer_id, role=role, content=content))
        session.commit()
    except Exception as e:
        # If message save fails, rollback and log the error
        session.rollback()
        print(f"Error saving message for customer_id={customer_id}: {e}")
        raise

def load_history(session, customer_id: str, limit: Optional[int] = None):
    q = session.execute(
        select(Message.role, Message.content)
        .where(Message.customer_id == customer_id)
        .order_by(Message.ts)
    )
    rows = q.all()
    if limit is not None:
        rows = rows[-limit:]
    return [{"role": r, "content": c} for (r, c) in rows]

def create_reservation(
    session,
    *,
    customer_id: Optional[str] = None,
    date: Optional[datetime.date] = None,
    time_start: Optional[datetime.time] = None,
    time_end: Optional[datetime.time] = None,
    start_iso: Optional[datetime] = None,
    end_iso: Optional[datetime] = None,
    party_size: Optional[int] = None,
    calendar_id: Optional[str] = None,
    event_id: Optional[str] = None,
    status: str = "confirmed",
    name: Optional[str] = None,
    store_id: Optional[int] = None,
    reservation_time: Optional[str] = None,
    special_requests: Optional[str] = None,
    waiter_id: Optional[int] = None,
    table_num: Optional[int] = None,
    tip: Optional[float] = None,
    total_ticket: Optional[float] = None,
    payment_method: Optional[str] = None,
):
    r = Reservation(
        customer_id=int(customer_id) if customer_id else None,
        name=name,
        date=date,
        time_start=time_start,
        time_end=time_end,
        start_iso=start_iso,
        end_iso=end_iso,
        party_size=party_size,
        calendar_id=calendar_id,
        event_id=event_id,
        status=status,
        store_id=store_id,
        reservation_time=reservation_time,
        special_requests=special_requests,
        waiter_id=waiter_id,
        table_num=table_num,
        tip=tip,
        total_ticket=total_ticket,
        payment_method=payment_method,
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

def list_reservations_by_customer_id(session, customer_id: str, limit: int = 50):
    try:
        customer_id_int = int(customer_id)
    except (ValueError, TypeError):
        customer_id_int = None
    
    return (session.query(Reservation)
            .filter_by(customer_id=customer_id_int)
            .order_by(desc(Reservation.created_at))
            .limit(limit)
            .all())

def get_last_event_id_by_customer_id(
    session,
    customer_id: str,
    require_confirmed: bool = True
) -> Optional[str]:
    try:
        customer_id_int = int(customer_id)
    except (ValueError, TypeError):
        customer_id_int = None
        
    base_q = session.query(Reservation.event_id).filter(
        Reservation.customer_id == customer_id_int,
        Reservation.event_id.isnot(None)
    )

    if require_confirmed:
        q = base_q.filter(Reservation.status == "confirmed")
        rec = q.order_by(desc(Reservation.updated_at), desc(Reservation.created_at)).first()
        if rec and rec[0]:
            return rec[0]

    rec = base_q.order_by(desc(Reservation.updated_at), desc(Reservation.created_at)).first()
    return rec[0] if rec and rec[0] else None

def upsert_user(session, *, customer_id, whatsapp: str | None = None, first_name: str | None = None, last_name: str | None = None):
    """Crea el usuario si no existe. Si existe y algunos campos están vacíos, los completa.
       No hace override de valores ya existentes (solo rellena None/'' )."""
    u = session.query(User).filter_by(customer_id=customer_id).one_or_none()
    if not u:
        u = User(customer_id=customer_id, whatsapp=whatsapp, first_name=first_name, last_name=last_name)
        session.add(u)
        try:
            session.commit()
            print(f"✅ Created user with customer_id={customer_id}, first_name={first_name}, last_name={last_name}")
        except IntegrityError:
            session.rollback()
            # carrera rara: otro proceso insertó; vuelve a leer
            u = session.query(User).filter_by(customer_id=customer_id).one()
            print(f"⚠️  User already exists, retrieved: customer_id={customer_id}")
    else:
        updated = False
        if (not u.whatsapp) and whatsapp:
            u.whatsapp = whatsapp
            updated = True
        if (not u.first_name) and first_name:
            u.first_name = first_name
            updated = True
        if (not u.last_name) and last_name:
            u.last_name = last_name
            updated = True
        if updated:
            session.commit()
            print(f"✅ Updated user with customer_id={customer_id}, first_name={first_name}, last_name={last_name}")
    return u
