from menu_index import load_menu_vector
from graph import build_app, GlobalState
from db import init_db, SessionLocal,save_message, load_history

init_db()
db = SessionLocal()

# Cargar el vector store del menú
vector_store = load_menu_vector(r"C:\Users\cabal\Desktop\Solertia\Solertia_MVP\menu_casona_completo.json")

# Construir el grafo/agent
app = build_app(vector_store)

if __name__ == "__main__":
    thread_id = "7226443592"
    
    state: GlobalState = {
        "thread_id": thread_id,
        "question": "",
        "messages": load_history(db,thread_id=thread_id),
        "context": [],
        "reservation_data": {},
        "answer": "",
        "route": "",
        "pending_reservation": False,  # <-- nuevo
    }


    print(" Asistente de La Casona — escribe 'salir' para terminar.\n")

    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("👋 ¡Hasta luego!")
            break
        save_message(db, thread_id, "user", user_input)

        # Actualizar la pregunta en el estado
        state["question"] = user_input

        # Pasar el estado actual al grafo
        state = app.invoke(state, config={"configurable": {"thread_id": thread_id}})

        # Mostrar la respuesta del agente
        print(f"Asistente: {state['answer']}")
        save_message(db, thread_id, "assistant", state["answer"])
