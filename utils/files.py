# utils/files.py
import hashlib
from pathlib import Path
import magic  # requires `python-magic`

ALLOWED_MIME = {"application/pdf"}

def hashed_name(original_name: str, data: bytes, suffix: str = ".pdf") -> str:
    """
    Deterministic filename based on content hash to avoid dupes.
    Produces: <original-stem>__<12-char-sha256><suffix>
    """
    stem = Path(original_name).stem
    digest = hashlib.sha256(data).hexdigest()[:12]
    safe_stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in stem).strip().replace(" ", "_")
    if not safe_stem:
        safe_stem = "file"
    return f"{safe_stem}__{digest}{suffix}"

def sniff_mime(data: bytes) -> str:
    return magic.from_buffer(data, mime=True) or ""

def too_big(num_bytes: int, max_mb: int) -> bool:
    return num_bytes > max_mb * 1024 * 1024