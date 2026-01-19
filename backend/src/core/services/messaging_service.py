import os
from src.core.graph import build_app
from src.core.menu_index import load_menu_vector
from src.core.db import SessionLocal, save_message, load_history
import re
import unicodedata
from pathlib import Path

# Load menu once (env override allowed)
MENU_URL = os.getenv("MENU_URL")  # production: Supabase public URL

# fallback local file for dev (backend/menu_casona_completo.json)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/core/services -> src/core -> src -> backend root
LOCAL_MENU_PATH = str(PROJECT_ROOT / "menu_casona_completo.json")

menu_source = MENU_URL or os.getenv("MENU_PATH") or LOCAL_MENU_PATH
vector_store = load_menu_vector(menu_source)
chatbot = build_app(vector_store)


# Very simple in-memory session store
_sessions = {}

def handle_whatsapp_message(incoming_msg: str, from_number: str) -> str:
    db = SessionLocal()
    thread_id = from_number
    
    numeric_id = ''.join(filter(str.isdigit, from_number))

    state = _sessions.get(thread_id) or {
        "thread_id": thread_id,
        "customer_id": numeric_id,  # ✅ inject numeric customer_id here
        "question": incoming_msg,
        "messages": load_history(db, customer_id=thread_id),
        "context": [],
        "reservation_data": {},
        "answer": "",
        "route": "",
        "pending_reservation": False,
        "pending_update": False,
        "human_handoff": False,
        "human_confirmation": False,
    }
    state["question"] = incoming_msg

    try:
        # Strip non-digit characters to keep numeric customer_id
        numeric_id = ''.join(filter(str.isdigit, thread_id))
        save_message(db, numeric_id, "user", incoming_msg)

        # ------------------------------------------------------------
        # HUMAN HANDOFF LOGIC (confirmation + filtering)
        # ------------------------------------------------------------
        text = incoming_msg.lower().strip()

        # Regex patterns to detect genuine human-contact requests
        HUMAN_KEYWORDS = [
            r"\bhablar con (un|una)? ?(humano|persona real|persona)\b",
            r"\bagente\b",
            r"\bpersona real\b",
            r"\batencion humana\b",
            r"\bcontactar con (un|una)? ?(humano|persona)\b",
        ]

        # --- Case 1: user is already in human mode → stay silent
        if state.get("human_handoff", False):
            _sessions[thread_id] = state
            return ""


        def normalize_text(t: str) -> str:
            """Lowercase, strip, remove accents and trim spaces."""
            t = t.lower().strip()
            t = ''.join(
                c for c in unicodedata.normalize('NFD', t)
                if unicodedata.category(c) != 'Mn'
            )
            return t

        # --- Case 2: bot is waiting for a yes/no confirmation
        if state.get("handoff_confirmation", False):
            clean = normalize_text(text)

            # Looser yes/no detection (accepts whole phrases)
            YES_PATTERNS = [
                r"\bsi\b", r"\bsii+\b", r"\bclaro\b", r"\bpor supuesto\b",
                r"\bobvio\b", r"\bquiero\b", r"\bseguro\b", r"\bde acuerdo\b"
            ]
            NO_PATTERNS = [
                r"\bno\b", r"\bnel\b", r"\bnah\b", r"\bno quiero\b",
                r"\bmejor no\b", r"\bcreo que no\b", r"\bno gracias\b"
            ]

            if any(re.search(p, clean) for p in YES_PATTERNS):
                state["human_handoff"] = True
                state["handoff_confirmation"] = False
                _sessions[thread_id] = state
                return "✅ Estás siendo conectado con un agente. Por favor, espera un momento."

            elif any(re.search(p, clean) for p in NO_PATTERNS):
                state["handoff_confirmation"] = False
                _sessions[thread_id] = state
                return "Perfecto 😊. ¿En qué puedo ayudarte?"

            else:
                return "Responde con un *Sí* o *No*."

        # --- Case 3: detect phrases requesting a human
        if any(re.search(p, text) for p in HUMAN_KEYWORDS):
            # Filter out numerical / plural contexts (e.g. "3 personas", "para 2 personas")
            if re.search(r"\d+\s*personas?", text) or re.search(r"\bpara\s+\d+\s*personas?\b", text):
                pass  # ignore, reservation context
            else:
                state["handoff_confirmation"] = True
                _sessions[thread_id] = state
                return "¿Estás seguro de que deseas hablar con un agente humano?"

        state = chatbot.invoke(state, config={"configurable": {"thread_id": thread_id}})
        answer = state.get("answer", "Lo siento, no entendí tu mensaje.")
        save_message(db, thread_id, "assistant", answer)
        _sessions[thread_id] = state
        return answer
    except Exception as e:
        return f"Ha ocurrido un error: {e}"
    finally:
        db.close()