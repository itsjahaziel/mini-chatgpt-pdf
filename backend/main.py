# backend/main.py
import os
import io
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pathlib import Path
import hashlib

from utils.embedder import get_or_create_collection, add_chunks
from utils.pdf_loader import extract_text_from_pdf
from utils.doc_loader import extract_text_from_docx, extract_text_from_txt_md
from utils.security import validate_upload
from utils.llm import answer_with_context

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="MiniChatGPT Backend", version="0.1.0")

# Allow local dev UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UPLOADS = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

collection = get_or_create_collection("pdf_chunks")


class ExtractResponse(BaseModel):
    filename: str
    saved_as: str
    bytes: int
    text_preview: str
    truncated: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    # Security & validation
    data = await file.read()
    ok, msg = validate_upload(
        filename=file.filename,
        data=data,
        max_bytes=5 * 1024 * 1024,  # 5MB limit
        allowed_exts={".pdf", ".docx", ".txt", ".md"},
    )
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # Stable hashed filename
    stem = Path(file.filename).stem
    digest = hashlib.sha256(data).hexdigest()[:12]
    ext = Path(file.filename).suffix.lower()
    final_name = f"{stem}__{digest}{ext}"
    final_path = UPLOADS / final_name

    if not final_path.exists():
        with open(final_path, "wb") as f:
            f.write(data)

    # Text extraction by type
    text = ""
    if ext == ".pdf":
        text = extract_text_from_pdf(str(final_path))
    elif ext == ".docx":
        text = extract_text_from_docx(io.BytesIO(data))
    else:  # .txt / .md
        text = extract_text_from_txt_md(io.BytesIO(data))

    preview = text[:2000]
    return ExtractResponse(
        filename=file.filename,
        saved_as=final_name,
        bytes=len(data),
        text_preview=preview,
        truncated=len(text) > len(preview),
    )


class IndexRequest(BaseModel):
    saved_names: List[str]
    tokens_per_chunk: int = 900
    overlap: int = 120


class IndexResponse(BaseModel):
    total_chunks: int
    indexed_files: int


@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest):
    # Read text from disk per saved_name, chunk, upsert
    from utils.chunker import chunk_text
    total = 0
    seen_files = 0
    for saved in req.saved_names:
        path = UPLOADS / saved
        if not path.exists():
            # skip silently
            continue

        ext = path.suffix.lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(str(path))
        elif ext == ".docx":
            text = extract_text_from_docx(path.open("rb"))
        else:
            text = extract_text_from_txt_md(path.open("rb"))

        if not text.strip():
            continue

        chunks = chunk_text(text, tokens_per_chunk=req.tokens_per_chunk, overlap=req.overlap)
        ids = [f"{saved}:{i}" for i in range(len(chunks))]
        metas = [{"source": saved, "chunk_id": i} for i in range(len(chunks))]
        add_chunks(collection, ids=ids, docs=chunks, metadatas=metas)
        total += len(chunks)
        seen_files += 1

    return IndexResponse(total_chunks=total, indexed_files=seen_files)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    sources: Optional[List[str]] = None
    model: str = "gpt-3.5-turbo"


class QueryResponse(BaseModel):
    matches: List[Dict[str, Any]]
    answer: str


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    where = None
    if req.sources:
        where = {"source": {"$in": req.sources}}

    try:
        res = collection.query(query_texts=[req.question], n_results=req.top_k, where=where)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    if not docs:
        return QueryResponse(matches=[], answer="I don’t know.")

    answer = answer_with_context(question=req.question, context_docs=docs, model=req.model)
    matches = []
    for d, m in zip(docs, metas):
        matches.append({"source": m.get("source"), "chunk_id": m.get("chunk_id"), "text": d[:1200]})

    return QueryResponse(matches=matches, answer=answer)