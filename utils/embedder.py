# utils/embedder.py
import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma")
ALLOW_RESET = os.getenv("ALLOW_CHROMA_RESET", "false").lower() == "true"

def _new_client() -> chromadb.PersistentClient:
    """Create a Chroma persistent client; optionally self-heal on schema mismatch."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=PERSIST_DIR)
    except Exception:
        if ALLOW_RESET:
            # Only nuke if you set ALLOW_CHROMA_RESET=true in .env (safety!)
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
            os.makedirs(PERSIST_DIR, exist_ok=True)
            return chromadb.PersistentClient(path=PERSIST_DIR)
        raise  # bubble up so you see the real error

def get_or_create_collection(name: str = "pdf_chunks"):
    """Return a persistent Chroma collection configured with OpenAI embeddings."""
    client = _new_client()
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    try:
        return client.get_collection(name=name, embedding_function=embed_fn)
    except Exception:
        return client.create_collection(name=name, embedding_function=embed_fn)

def add_chunks(collection, ids: List[str], docs: List[str], metadatas: List[Dict[str, Any]]):
    """Add/Upsert documents into the Chroma collection."""
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    else:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)