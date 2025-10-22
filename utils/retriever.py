from typing import Tuple, List, Dict, Any

def search_chunks(collection, query: str, k: int = 4) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Query similar chunks from Chroma.
    Returns (documents, metadatas).
    """
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas