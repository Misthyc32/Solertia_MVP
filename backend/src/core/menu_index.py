import json
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def load_menu_vector(path_or_url: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = InMemoryVectorStore(embeddings)

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        r = httpx.get(path_or_url, timeout=30.0)
        r.raise_for_status()
        menu = r.json()
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            menu = json.load(f)

    docs = []
    for categoria, platillos in menu.items():
        for platillo in platillos:
            content = (
                f"{platillo['nombre']} ({categoria}): "
                f"{platillo.get('ingredientes', 'Sin ingredientes')}. "
                f"Precio: ${platillo['precio']}"
            )
            docs.append(Document(page_content=content, metadata={"categoria": categoria}))

    vs.add_documents(docs)
    return vs
