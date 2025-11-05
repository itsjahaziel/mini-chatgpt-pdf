import os
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import extract_text_from_pdf, get_pdf_info
from utils.chunker import chunk_text
from utils.embedder import get_or_create_collection, add_chunks
from utils.llm import answer_with_context
from utils.files import hashed_name, sniff_mime, too_big, ALLOWED_MIME, human_bytes

# ---------- env / page ----------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # loads OPENAI_API_KEY, MAX_FILE_MB, MAX_PAGES

st.set_page_config(page_title="Mini ChatGPT for PDFs", page_icon="📄", layout="wide")
st.title("📄 Mini ChatGPT for PDFs")

# ---------- simple styles ----------
st.markdown(
    """
    <style>
      .card { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:16px; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:12px; margin-right:6px; }
      .pill-ok { background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; }
      .pill-warn { background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; }
      .pill-info { background:#EFF6FF; color:#1E40AF; border:1px solid #BFDBFE; }
      .file-header { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
      .file-name { font-weight:700; color:#0F172A; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- state ----------
if "texts" not in st.session_state:
    st.session_state["texts"]: Dict[str, str] = {}          # filename -> extracted text
if "indexed_sources" not in st.session_state:
    st.session_state["indexed_sources"]: List[str] = []     # which filenames have embeddings
if "file_meta" not in st.session_state:
    # filename -> {'bytes': int, 'mime': str, 'pages_total': int, 'pages_used': int, 'title': Optional[str]}
    st.session_state["file_meta"]: Dict[str, Dict[str, Any]] = {}

# ---------- helpers ----------
def list_indexed_sources(collection) -> List[str]:
    """Return unique 'source' values stored in Chroma metadatas (paginated & robust)."""
    sources = set()
    try:
        offset, page = 0, 1000
        while True:
            batch = collection.get(include=["metadatas"], limit=page, offset=offset)
            metadatas = batch.get("metadatas") or []
            if not metadatas:
                break

            items = []
            if len(metadatas) and isinstance(metadatas[0], dict):
                items = metadatas
            else:
                for rowlist in metadatas:
                    items.extend(rowlist or [])

            for md in items:
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
    """Return [{'source': name, 'chunks': count}, ...] with pagination."""
    counts: Dict[str, int] = {}
    try:
        offset, page = 0, 1000
        while True:
            batch = collection.get(include=["metadatas"], limit=page, offset=offset)
            metadatas = batch.get("metadatas") or []
            if not metadatas:
                break

            items = []
            if len(metadatas) and isinstance(metadatas[0], dict):
                items = metadatas
            else:
                for rowlist in metadatas:
                    items.extend(rowlist or [])

            for md in items:
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


def index_files(collection, filenames: List[str], tokens_per_chunk: int, overlap: int) -> int:
    """Index (or re-index) the given filenames using stable IDs; returns total chunks indexed."""
    total = 0
    for fname in filenames:
        text = st.session_state["texts"].get(fname, "")
        if not text:
            continue
        chunks = chunk_text(text, tokens_per_chunk=tokens_per_chunk, overlap=overlap)
        ids = [f"{fname}:{i}" for i in range(len(chunks))]  # stable per-file IDs
        metas = [{"source": fname, "chunk_id": i} for i in range(len(chunks))]
        try:
            add_chunks(collection, ids=ids, docs=chunks, metadatas=metas)
            total += len(chunks)
            if fname not in st.session_state["indexed_sources"]:
                st.session_state["indexed_sources"].append(fname)
        except Exception as e:
            st.error(f"{fname}: indexing failed → {e}")
    return total


# ---------- sidebar: knobs & index maintenance ----------
st.sidebar.header("Settings")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.sidebar.warning("Set OPENAI_API_KEY in .env for embeddings & answers.")
else:
    st.sidebar.success("OpenAI API key detected.")

tokens_per_chunk = st.sidebar.slider("Tokens per chunk", 300, 1500, 900, 50)
overlap = st.sidebar.slider("Token overlap", 0, 300, 120, 10)
top_k = st.sidebar.slider("Top-k chunks", 1, 8, 4, 1)
model_name = st.sidebar.selectbox("LLM model", ["gpt-3.5-turbo"], index=0)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset ALL embeddings"):
    from shutil import rmtree
    try:
        rmtree("data/processed/chroma")
        st.sidebar.success("All embeddings deleted.")
        st.session_state["indexed_sources"] = []
    except FileNotFoundError:
        st.sidebar.info("No index on disk.")

# ---------- layout ----------
col_ingest, col_index = st.columns([2, 1], gap="large")

# ==============================
# 1) INGEST (safe upload + page-limit guard)
# ==============================
with col_ingest:
    st.subheader("1) Upload & Extract")

    MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "5"))
    MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))
    st.caption(
        f"Max file size: {MAX_FILE_MB} MB • Max pages read: {MAX_PAGES} • "
        f"Allowed types: {', '.join(sorted(ALLOWED_MIME))}"
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs (text-based, not scans)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        uploads_dir = ROOT / "data" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        for uf in uploaded_files:
            data = uf.getvalue()

            # size check
            if too_big(len(data), MAX_FILE_MB):
                st.warning(f"Skipped **{uf.name}**: over {MAX_FILE_MB} MB.")
                continue

            # MIME check
            mime = sniff_mime(data)
            if mime not in ALLOWED_MIME:
                st.warning(f"Skipped **{uf.name}**: unsupported type `{mime}`.")
                continue

            final_name = hashed_name(uf.name, data, suffix=".pdf")
            final_path = uploads_dir / final_name

            if final_name in st.session_state["texts"]:
                st.info(f"Duplicate upload skipped (already extracted): {final_name}")
                continue

            if not final_path.exists():
                final_path.write_bytes(data)

            try:
                pages_total, title = get_pdf_info(str(final_path))
            except Exception as e:
                st.error(f"Failed to read metadata for {final_name}: {e}")
                pages_total, title = 0, None

            try:
                # Page-limit guard: only read the first MAX_PAGES pages
                text = extract_text_from_pdf(str(final_path), max_pages=MAX_PAGES)
                if text and text.strip():
                    st.session_state["texts"][final_name] = text
                    st.session_state["file_meta"][final_name] = {
                        "bytes": len(data),
                        "mime": mime,
                        "pages_total": pages_total,
                        "pages_used": min(pages_total, MAX_PAGES),
                        "title": title,
                    }
                    truncated = pages_total > MAX_PAGES
                    msg = "Saved & Extracted"
                    if truncated:
                        msg += f" (truncated to first {MAX_PAGES} pages)"
                    st.success(f"{msg}: {final_name}")
                else:
                    st.warning(f"No text found in: {final_name} (likely a scanned PDF)")
            except Exception as e:
                st.error(f"Failed to extract {final_name}: {e}")

    # previews + meta badges
    if st.session_state["texts"]:
        st.markdown("**Extracted files**")
        for fname in sorted(st.session_state["texts"].keys()):
            meta = st.session_state["file_meta"].get(fname, {})
            size_pill = f'<span class="pill pill-info">{human_bytes(meta.get("bytes", 0))}</span>' if meta else ""
            mime_pill = f'<span class="pill pill-info">{meta.get("mime","?")}</span>' if meta else ""
            pages_used = meta.get("pages_used")
            pages_total = meta.get("pages_total")
            if pages_used and pages_total:
                pages_label = f"{pages_used}/{pages_total} pages" if pages_total >= pages_used else f"{pages_used} pages"
            elif pages_total:
                pages_label = f"{pages_total} pages"
            else:
                pages_label = "pages: ?"

            truncated = (pages_total or 0) > pages_used if (pages_total and pages_used) else False
            pages_pill = (
                f'<span class="pill {"pill-warn" if truncated else "pill-ok"}">{pages_label}'
                f'{" • truncated" if truncated else ""}</span>'
            )

            title = meta.get("title")
            title_pill = f'<span class="pill pill-ok">title: {title}</span>' if title else ""

            header_html = (
                f'<div class="file-header">'
                f'<span class="file-name">{fname}</span>'
                f'{size_pill}{mime_pill}{pages_pill}{title_pill}'
                f'</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)
            with st.expander("Preview text", expanded=False):
                txt = st.session_state["texts"][fname]
                st.text(txt[:2000] + ("..." if len(txt) > 2000 else ""))

# ==============================
# 2) INDEX (build & manage; multi-select; per-file delete; stats)
# ==============================
with col_index:
    st.subheader("2) Build / Manage Index")
    collection = get_or_create_collection("pdf_chunks")

    extracted_names = sorted(st.session_state["texts"].keys())
    to_index = st.multiselect(
        "Select files to (re)index",
        options=extracted_names,
        default=extracted_names,
        help="You can re-index a file; IDs are stable, so duplicates are avoided.",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📦 Create / Update index (selected)"):
            if not api_key:
                st.error("Set OPENAI_API_KEY in .env.")
            elif not to_index:
                st.info("Select at least one file.")
            else:
                total = index_files(collection, to_index, tokens_per_chunk, overlap)
                if total:
                    st.success(f"Indexed/updated {total} chunks across {len(to_index)} file(s).")
    with c2:
        if st.button("🔄 Rebuild index from ALL extracted files"):
            if not api_key:
                st.error("Set OPENAI_API_KEY in .env.")
            elif not extracted_names:
                st.info("No extracted files yet.")
            else:
                total = index_files(collection, extracted_names, tokens_per_chunk, overlap)
                if total:
                    st.success(f"Rebuilt/updated {total} chunks across {len(extracted_names)} file(s).")

    # per-source delete + stats
    indexed_sources = list_indexed_sources(collection)
    if indexed_sources:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Indexed sources on disk**")
        del_choice = st.selectbox("Delete embeddings for a file", ["—"] + indexed_sources)
        if st.button("🗑️ Delete selected embeddings") and del_choice != "—":
            try:
                collection.delete(where={"source": del_choice})
                st.success(f"Deleted embeddings for: {del_choice}")
                if del_choice in st.session_state["indexed_sources"]:
                    st.session_state["indexed_sources"].remove(del_choice)
            except Exception as e:
                st.error(f"Delete failed: {e}")

        stats = get_index_stats(collection)
        if stats:
            st.markdown("**Index summary**")
            st.table(stats)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No indexed sources yet. Build the index after extracting.")

# ==============================
# 3) QUESTION ANSWERING
# ==============================
st.markdown("---")
st.subheader("3) Ask Questions (grounded in your PDFs)")

indexed_sources = list_indexed_sources(collection)
if indexed_sources:
    source_filter = st.multiselect(
        "Limit search to these files (optional)",
        options=indexed_sources,
        default=indexed_sources,
    )
else:
    source_filter = []

question = st.text_input("Your question")
go = st.button("🔎 Search & Answer")

if go:
    if not question.strip():
        st.error("Type a question.")
    elif not api_key:
        st.error("Set OPENAI_API_KEY in .env.")
    else:
        # metadata filter
        where = None
        if source_filter and len(source_filter) != len(indexed_sources):
            where = {"source": {"$in": source_filter}}

        try:
            res = collection.query(
                query_texts=[question],
                n_results=top_k,
                where=where,
            )
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
        except Exception as e:
            st.error(f"Search failed: {e}")
            docs, metas = [], []

        if not docs:
            st.warning("No relevant chunks found. Try a more specific question or adjust the source filter.")
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