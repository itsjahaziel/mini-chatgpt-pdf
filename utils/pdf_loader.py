# utils/pdf_loader.py
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    """
    Try text extraction via PyMuPDF; if almost no text, fallback to OCR.
    """
    text_parts = []
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
    except Exception:
        text_parts = []

    text = "\n".join(text_parts).strip()
    if len(text) >= 100:
        return text

    # Fallback to OCR for scanned PDFs
    try:
        from utils.ocr import ocr_pdf_to_text
        ocr_text = ocr_pdf_to_text(file_path)
        return ocr_text.strip()
    except Exception:
        # If OCR fails, just return whatever we had
        return text