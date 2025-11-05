# utils/doc_loader.py
from io import BytesIO
from typing import Union

def extract_text_from_docx(source: Union[BytesIO, bytes]):
    try:
        from docx import Document
    except Exception:
        return ""

    if isinstance(source, bytes):
        source = BytesIO(source)

    doc = Document(source)
    parts = []
    for p in doc.paragraphs:
        parts.append(p.text)
    return "\n".join(parts)

def extract_text_from_txt_md(source: Union[BytesIO, bytes]):
    if isinstance(source, bytes):
        source = BytesIO(source)
    try:
        text = source.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return text