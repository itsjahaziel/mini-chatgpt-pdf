# utils/embedder.py

import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

_PERSIST_DIR = "data/processed/chroma"

def _new_client():
    """Create a Chroma persistent client; auto-repair the folder on schema mismatch."""
    os.makedirs(_PERSIST_DIR, exist_ok=True)
    try:
        return chromadb.PersistentClient(path=_PERSIST_DIR)
    except Exception as e:
        # Common when upgrading Chroma: local db schema mismatch (e.g., "no such table: databases")
        # Self-heal by nuking the directory once, then re-create.
        try:
            shutil.rmtree(_PERSIST_DIR, ignore_errors=True)
            os.makedirs(_PERSIST_DIR, exist_ok=True)
            return chromadb.PersistentClient(path=_PERSIST_DIR)
        except Exception:
            # Bubble up if we truly cannot init
            raise

def get_or_create_collection(name: str = "pdf_chunks"):
    """
    Return a persistent Chroma collection configured with OpenAI embeddings.
    Auto-repairs the persistence directory if initialization fails.
    """
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
