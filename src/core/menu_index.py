import json
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def load_menu_vector(path: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = InMemoryVectorStore(embeddings)
    with open(path, "r", encoding="utf-8") as f:
        menu = json.load(f)
    docs = []
    for categoria, platillos in menu.items():
        for platillo in platillos:
            content = f"{platillo['nombre']} ({categoria}): {platillo.get('ingredientes', 'Sin ingredientes')}. Precio: ${platillo['precio']}"
            docs.append(Document(page_content=content, metadata={"categoria": categoria}))
    vs.add_documents(docs)
    return vs
