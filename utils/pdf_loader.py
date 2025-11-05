# utils/pdf_loader.py
import os
from typing import List, Tuple
import fitz  # PyMuPDF

USE_OCR = os.getenv("USE_OCR", "false").lower() == "true"

if USE_OCR:
    from pdf2image import convert_from_path
    import pytesseract

def extract_text_from_pdf(file_path: str, min_text_len: int = 100) -> Tuple[str, bool]:
    """
    Extract text from a PDF. Returns (text, used_ocr).
    Tries fast native text first, then falls back to OCR if enabled.
    """
    parts: List[str] = []
    used_ocr = False

    # 1) Fast path: native text
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                parts.append(page.get_text())
        text = "\n".join(parts).strip()
        if len(text) >= min_text_len or not USE_OCR:
            return text, False
    except Exception:
        # fall through to OCR if allowed
        pass

    # 2) OCR fallback
    if USE_OCR:
        images = convert_from_path(file_path)  # requires poppler
        ocr_text = []
        for img in images:
            ocr_text.append(pytesseract.image_to_string(img))
        text = "\n\n".join(ocr_text).strip()
        used_ocr = True
        return text, used_ocr

    # No text and OCR disabled
    return "", False