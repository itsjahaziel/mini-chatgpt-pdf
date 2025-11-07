# utils/security.py
import io
import magic

def validate_upload(filename: str, data: bytes, max_bytes: int, allowed_exts: set[str]) -> tuple[bool, str]:
    if not filename:
        return False, "Missing filename."

    if len(data) == 0:
        return False, "Empty file."

    if len(data) > max_bytes:
        return False, f"File too large. Limit is {max_bytes} bytes."

    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in allowed_exts:
        return False, f"Unsupported file type: {ext}"

    # MIME sniff (best-effort)
    try:
        mime = magic.from_buffer(data, mime=True)
        if ext == ".pdf" and mime not in {"application/pdf"}:
            return False, f"Invalid PDF (MIME: {mime})"
        if ext == ".docx" and mime not in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }:
            return False, f"Invalid DOCX (MIME: {mime})"
        if ext in {".txt", ".md"} and not mime.startswith(("text/", "application/octet-stream")):
            return False, f"Invalid text file (MIME: {mime})"
    except Exception:
        # If magic fails, still allow based on ext + size
        pass

    return True, "ok"