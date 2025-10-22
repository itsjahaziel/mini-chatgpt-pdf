import fitz  # PyMuPDF
from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    text_parts: List[str] = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)