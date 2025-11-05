# utils/ocr.py
from typing import List
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf_to_text(file_path: str) -> str:
    """
    Convert each PDF page to image, run Tesseract OCR, return concatenated text.
    Requires: poppler (for pdf2image) and tesseract installed on system.
    """
    pages: List[str] = []
    images = convert_from_path(file_path)  # may raise if poppler not installed
    for img in images:
        txt = pytesseract.image_to_string(img)
        if txt:
            pages.append(txt.strip())
    return "\n\n".join(pages)