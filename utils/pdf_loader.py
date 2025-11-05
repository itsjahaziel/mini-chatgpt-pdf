# utils/pdf_loader.py
from typing import Tuple, Optional, List
import fitz  # PyMuPDF


def get_pdf_info(file_path: str) -> Tuple[int, Optional[str]]:
    """
    Return (page_count, metadata_title_or_None).
    Keeps it fast: doesn't read page text.
    """
    with fitz.open(file_path) as doc:
        page_count = doc.page_count
        title = (doc.metadata or {}).get("title") or None
        return page_count, title


def extract_text_from_pdf(file_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF with an optional page cap.
    If max_pages is set, only the first `max_pages` pages are read.
    """
    parts: List[str] = []
    with fitz.open(file_path) as doc:
        last = min(doc.page_count, max_pages) if max_pages else doc.page_count
        for i in range(last):
            parts.append(doc[i].get_text())
    return "\n".join(parts)