# app.py
import os
import re
import io
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import requests
from dotenv import load_dotenv

from utils.pdf_loader import extract_text_from_pdf
from utils.doc_loader import extract_text_from_docx, extract_text_from_txt_md
from utils.chunker import chunk_text
from utils.embedder import get_or_create_collection, add_chunks
from utils.llm import answer_with_context

# ---------- env / page ----------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Mini ChatGPT for PDFs", page_icon="📄", layout="wide")
st.title("📄 Mini ChatGPT for PDFs (V2)")

BACKEND_URL = os.getenv("BACKEND_URL", "").strip()

# ---------- styles ----------
st.markdown(
    """
    <style>
      .card { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:16px; }
      .muted { color:#6B7280; }
      .tight { margin-top: .25rem; }
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#ECFDF5; color:#065F46; font-weight:600; font-size:12px; }
      .warnpill { display:inline-block; padding:2px 8px; border-radius:999px; background:#FEF3C7; color:#92400E; font-weight:600; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- state ----------
if "texts" not in st.session_state:
    st.session_state["texts"]: Dict[str, str] = {}         # filename -> extracted text
if "saved_names" not in st.session_state:
    st.session_state["saved_names"]: List[str] = []        # filenames stored on disk (hashed)
if "history" not in st.session_state:
    st.session_state["history"]: List[Dict] = []           # Q&A history

# ---------- helpers ----------
def sanitize_filename(name: str) -> str:
    base, ext = os.path.splitext(name)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return f"{base}{ext or '.pdf'}"

def hashed_name(original_name: str, data: bytes) -> str:
    stem = Path(sanitize_filename(original_name)).stem
    digest = hashlib.sha256(data).hexdigest()[:12]
    ext = Path(original_name).suffix or ".pdf"
    return f"{stem}__{digest}{ext}"

def list_indexed_sources(collection) -> List[str]:
    sources = set()
    try:
        offset = 0
        page = 1000
        while True:
            batch = collection.get(include=["metadatas"], limit=page, offset=offset)
            metadatas = batch.get("metadatas") or []
            if not metadatas:
                break

            if len(metadatas) and isinstance(metadatas[0], dict):
                it = metadatas
            else:
                it = []
                for rowlist in metadatas:
                    it.extend(rowlist or [])

            for md in it:
                s = (md or {}).get("source")
                if s:
                    sources.add(s)

            ids = batch.get("ids") or []
            if len(ids) < page:
                break
            offset += page
    except Exception:
        pass
    return sorted(sources)

def get_index_stats(collection) -> List[dict]:
    counts: Dict[str, int] = {}
    try:
        offset = 0
        page = 1000
        while True:
            batch = collection.get(include=["metadatas"], limit=page, offset=offset)
            metadatas = batch.get("metadatas") or []
            if not metadatas:
                break

            if len(metadatas) and isinstance(metadatas[0], dict):
                it = metadatas
            else:
                it = []
                for rowlist in metadatas:
                    it.extend(rowlist or [])

            for md in it:
                s = (md or {}).get("source")
                if s:
                    counts[s] = counts.get(s, 0) + 1

            ids = batch.get("ids") or []
            if len(ids) < page:
                break
            offset += page
    except Exception:
        pass
    rows = [{"source": s, "chunks": n} for s, n in sorted(counts.items(), key=lambda x: x[0].lower())]
    return rows

def index_files_local(collection, filenames: List[str], tokens_per_chunk: int, overlap: int) -> int:
    total = 0
    uploads_dir = ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    for saved in filenames:
        path = uploads_dir / saved
        if not path.exists():
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

        chunks = chunk_text(text, tokens_per_chunk=tokens_per_chunk, overlap=overlap)
        ids = [f"{saved}:{i}" for i in range(len(chunks))]
        metas = [{"source": saved, "chunk_id": i} for i in range(len(chunks))]
        try:
            add_chunks(collection, ids=ids, docs=chunks, metadatas=metas)
            total += len(chunks)
        except Exception as e:
            st.error(f"{saved}: indexing failed → {e}")
    return total

# ---------- sidebar ----------
st.sidebar.header("Settings")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.sidebar.warning("Set OPENAI_API_KEY in .env for embeddings & answers.")
else:
    st.sidebar.success("OpenAI API key detected.")

if BACKEND_URL:
    st.sidebar.success(f"Backend: {BACKEND_URL}")
else:
    st.sidebar.info("Backend not set. Running in direct mode.")

tokens_per_chunk = st.sidebar.slider("Tokens per chunk", 300, 1500, 900, 50)
overlap = st.sidebar.slider("Token overlap", 0, 300, 120, 10)
top_k = st.sidebar.slider("Top-k chunks", 1, 8, 4, 1)
model_name = st.sidebar.selectbox("LLM model", ["gpt-3.5-turbo"], index=0)

# History export
st.sidebar.markdown("---")
if st.sidebar.button("⬇️ Export Q&A history (JSON)"):
    data = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2)
    st.sidebar.download_button("Download history.json", data=data, file_name="history.json", mime="application/json")

# Reset index
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset ALL embeddings"):
    from shutil import rmtree
    try:
        rmtree("data/processed/chroma")
        st.sidebar.success("All embeddings deleted.")
    except FileNotFoundError:
        st.sidebar.info("No index on disk.")

# ---------- layout ----------
col_ingest, col_index = st.columns([2, 1], gap="large")

# ==============================
# 1) INGEST (multi-file upload: PDF, DOCX, TXT, MD)
# ==============================
with col_ingest:
    st.subheader("1) Upload & Extract")
    with st.container():
        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            uploads_dir = ROOT / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            for uf in uploaded_files:
                data = uf.getvalue()
                final_name = hashed_name(uf.name, data)
                final_path = uploads_dir / final_name

                if BACKEND_URL:
                    # Send to backend for validation + extraction
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/extract",
                            files={"file": (uf.name, data, uf.type or "application/octet-stream")},
                            timeout=60,
                        )
                        if resp.status_code != 200:
                            st.error(f"{uf.name}: {resp.text}")
                            continue
                        js = resp.json()
                        saved_as = js["saved_as"]
                        st.session_state["saved_names"].append(saved_as)
                        # Store preview text locally for preview UX
                        st.session_state["texts"][saved_as] = js["text_preview"]
                        st.success(f"Saved & extracted (via backend): {saved_as}")
                    except Exception as e:
                        st.error(f"Backend extract failed for {uf.name}: {e}")
                else:
                    # Direct mode
                    if not final_path.exists():
                        with open(final_path, "wb") as f:
                            f.write(data)

                    ext = final_path.suffix.lower()
                    try:
                        if ext == ".pdf":
                            text = extract_text_from_pdf(str(final_path))
                        elif ext == ".docx":
                            text = extract_text_from_docx(io.BytesIO(data))
                        else:
                            text = extract_text_from_txt_md(io.BytesIO(data))
                        if text and text.strip():
                            st.session_state["texts"][final_name] = text
                            st.session_state["saved_names"].append(final_name)
                            st.success(f"Saved & extracted: {final_name}")
                        else:
                            st.warning(f"No text found in: {final_name} (check if scanned PDF)")
                    except Exception as e:
                        st.error(f"Failed to extract {final_name}: {e}")

    # Previews
    if st.session_state["texts"]:
        st.markdown("**Extracted files (preview)**")
        for fname, txt in sorted(st.session_state["texts"].items()):
            with st.expander(f"{fname} — preview", expanded=False):
                st.text(txt[:2000] + ("..." if len(txt) > 2000 else ""))

# ==============================
# 2) INDEX (build/manage)
# ==============================
with col_index:
    st.subheader("2) Build / Manage Index")
    collection = get_or_create_collection("pdf_chunks")

    saved_names = sorted(set(st.session_state["saved_names"]))
    to_index = st.multiselect(
        "Select files to (re)index",
        options=saved_names,
        default=saved_names,
        help="Stable per-file IDs avoid duplicate embeddings.",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📦 Create / Update index (selected)"):
            if not api_key:
                st.error("Set OPENAI_API_KEY in .env.")
            elif not to_index:
                st.info("Select at least one file.")
            else:
                if BACKEND_URL:
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/index",
                            json={
                                "saved_names": to_index,
                                "tokens_per_chunk": tokens_per_chunk,
                                "overlap": overlap,
                            },
                            timeout=300,
                        )
                        if resp.status_code != 200:
                            st.error(f"Backend index failed: {resp.text}")
                        else:
                            js = resp.json()
                            st.success(f"Indexed {js['total_chunks']} chunks across {js['indexed_files']} file(s).")
                    except Exception as e:
                        st.error(f"Backend index call failed: {e}")
                else:
                    total = index_files_local(collection, to_index, tokens_per_chunk, overlap)
                    if total:
                        st.success(f"Indexed/updated {total} chunks across {len(to_index)} file(s).")

    with c2:
        if st.button("🔄 Rebuild index from ALL uploaded"):
            if not api_key:
                st.error("Set OPENAI_API_KEY in .env.")
            elif not saved_names:
                st.info("No files uploaded yet.")
            else:
                if BACKEND_URL:
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/index",
                            json={
                                "saved_names": saved_names,
                                "tokens_per_chunk": tokens_per_chunk,
                                "overlap": overlap,
                            },
                            timeout=600,
                        )
                        if resp.status_code != 200:
                            st.error(f"Backend rebuild failed: {resp.text}")
                        else:
                            js = resp.json()
                            st.success(f"Rebuilt {js['total_chunks']} chunks across {js['indexed_files']} file(s).")
                    except Exception as e:
                        st.error(f"Backend rebuild call failed: {e}")
                else:
                    total = index_files_local(collection, saved_names, tokens_per_chunk, overlap)
                    if total:
                        st.success(f"Rebuilt/updated {total} chunks across {len(saved_names)} file(s).")

    # Index summary
    indexed_sources = []
    try:
        indexed_sources = list_indexed_sources(collection)
    except Exception:
        pass

    if indexed_sources:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Indexed sources on disk**")

        # stats table
        stats = get_index_stats(collection)
        if stats:
            st.table(stats)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No indexed sources yet. Build the index after extracting.")

# ==============================
# 3) QUESTION ANSWERING + History
# ==============================
st.markdown("---")
st.subheader("3) Ask Questions (grounded in your docs)")

source_filter = []
if indexed_sources:
    source_filter = st.multiselect(
        "Limit search to these files (optional)",
        options=indexed_sources,
        default=indexed_sources,
    )

question = st.text_input("Your question")
go = st.button("🔎 Search & Answer")

if go:
    if not question.strip():
        st.error("Type a question.")
    elif not api_key:
        st.error("Set OPENAI_API_KEY in .env.")
    else:
        if BACKEND_URL:
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "sources": source_filter if source_filter and len(source_filter) != 0 else None,
                        "model": model_name,
                    },
                    timeout=120,
                )
                if resp.status_code != 200:
                    st.error(f"Backend query failed: {resp.text}")
                else:
                    js = resp.json()
                    matches = js.get("matches", [])
                    if not matches:
                        st.warning("No relevant chunks found.")
                    else:
                        st.markdown("**Top matches**")
                        for i, m in enumerate(matches, start=1):
                            with st.expander(f"[{i}] {m['source']} · chunk {m['chunk_id']}", expanded=False):
                                st.text(m["text"])

                    st.markdown("### Answer")
                    st.write(js.get("answer", "I don’t know."))

                    # Save to history
                    st.session_state["history"].append(
                        {"q": question, "answer": js.get("answer", ""), "sources": [m["source"] for m in matches]}
                    )
            except Exception as e:
                st.error(f"Backend query call failed: {e}")
        else:
            # direct vector search
            collection = get_or_create_collection("pdf_chunks")
            where = None
            if source_filter and len(source_filter) != 0 and len(source_filter) != len(indexed_sources):
                where = {"source": {"$in": source_filter}}
            try:
                res = collection.query(query_texts=[question], n_results=top_k, where=where)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
            except Exception as e:
                st.error(f"Search failed: {e}")
                docs, metas = [], []

            if not docs:
                st.warning("No relevant chunks found.")
            else:
                st.markdown("**Top matches**")
                for i, (d, m) in enumerate(zip(docs, metas), start=1):
                    src = m.get("source", "?")
                    cid = m.get("chunk_id", "?")
                    with st.expander(f"[{i}] {src} · chunk {cid}", expanded=False):
                        st.text(d[:1600] + ("..." if len(d) > 1600 else ""))

                try:
                    answer = answer_with_context(question=question, context_docs=docs, model=model_name)
                except Exception as e:
                    st.error(f"Answer failed: {e}")
                else:
                    st.markdown("### Answer")
                    st.write(answer)
                    st.caption('If the answer is not in the context, the assistant will say "I don’t know."')
                    st.session_state["history"].append(
                        {"q": question, "answer": answer, "sources": [m.get("source") for m in metas]}
                    )

# ==============================
# 4) HISTORY PANEL
# ==============================
st.markdown("---")
st.subheader("History")
if st.session_state["history"]:
    for i, item in enumerate(st.session_state["history"], start=1):
        with st.expander(f"Q{i}: {item['q']}", expanded=False):
            st.markdown("**Answer**")
            st.write(item["answer"])
            if item.get("sources"):
                st.caption("Sources: " + ", ".join(item["sources"]))
else:
    st.caption("No history yet. Ask something!")