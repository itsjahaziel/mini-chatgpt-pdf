import tiktoken
from typing import List

def chunk_text(text: str, tokens_per_chunk: int = 900, overlap: int = 120) -> List[str]:
    """
    Token-aware chunking using cl100k_base (works with GPT-3.5/4 families).
    overlap keeps some context continuity between neighboring chunks.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    chunks: List[str] = []
    i = 0
    while i < len(token_ids):
        window = token_ids[i:i + tokens_per_chunk]
        chunks.append(enc.decode(window))
        if i + tokens_per_chunk >= len(token_ids):
            break
        step = max(1, tokens_per_chunk - overlap)
        i += step
    return chunks