# Mini ChatGPT for PDFs (Streamlit MVP)

**Upload → Extract → Chunk → Embed → Retrieve → Answer**  
Answers come **only** from your PDFs. If the info isn’t in the docs, the app replies **“I don’t know.”**

---

## ✨ Features
- **Multi-PDF upload** (text-based PDFs)
- **Text extraction** with PyMuPDF
- **Token-aware chunking** (configurable size & overlap)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector store**: Chroma (persistent on disk)
- **Top-k retrieval** + **GPT-3.5** for grounded answers
- **Source controls**: filter search by specific file(s)
- **Index tools**: per-file delete, “Rebuild index from ALL extracted files”
- **De-dup by content**: same file re-uploads are skipped
- **Index summary**: shows chunk counts per file

---

## 🚀 Quickstart

### Requirements
- Python **3.10+**
- An OpenAI API key

### Setup
```bash
git clone https://github.com/itsjahaziel/mini-chatgpt-pdf.git
cd mini-chatgpt-pdf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set:
# OPENAI_API_KEY=sk-...your key...

streamlit run app.py

🧭 Usage
	1.	Upload & Extract
Upload one or more text-based PDFs. The app extracts text and shows a preview.
	2.	Build / Manage Index
	•	Select the files to index and click Create / Update index
	•	Or click Rebuild index from ALL extracted files
	•	Use Delete embeddings for a file if you need to remove a single document from the index
	3.	Ask Questions
	•	Optionally limit search to specific file(s)
	•	Ask your question → the app retrieves top-k chunks and answers using only that context
	•	If it’s not in the docs, it responds “I don’t know.”

Tip: Ask specific questions for best results (dates, roles, definitions, etc.).

⸻

🧩 How it works (RAG pipeline)
	1.	Extract text from PDFs (PyMuPDF)
	2.	Chunk text into token-bounded pieces (size & overlap sliders)
	3.	Embed chunks with OpenAI text-embedding-3-small
	4.	Store vectors in Chroma (persistent at data/processed/chroma/)
	5.	Retrieve top-k similar chunks for a query
	6.	Answer with gpt-3.5-turbo using a strict prompt: use only the provided context; otherwise say “I don’t know.”

⸻

🔧 Configuration
	•	.env
OPENAI_API_KEY=sk-your-key-here

	•	In the app sidebar:
	•	Tokens per chunk, Token overlap
	•	Top-k retrieved chunks
	•	LLM model (default: gpt-3.5-turbo)

⸻

📁 Project structure

mini-chatgpt-pdf/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── utils/
│   ├── pdf_loader.py      # PyMuPDF extraction
│   ├── chunker.py         # token-aware chunking (tiktoken)
│   ├── embedder.py        # OpenAI embeddings + Chroma (persistent)
│   ├── retriever.py       # (kept minimal or unused in latest)
│   └── llm.py             # GPT-3.5 answer w/ strict context-only prompt
└── data/
    ├── uploads/           # saved PDFs (ignored by git)
    └── processed/chroma/  # vector DB (ignored by git)


🛡️ Security & privacy
	•	Never commit your real .env. This repo ignores it by default.
	•	PDFs are stored locally (data/uploads/) and embeddings locally (data/processed/chroma/).
	•	This is an MVP for demos—review before using on sensitive data.

⸻

🧹 Troubleshooting
	•	“I uploaded the same file and it duplicates”
The app de-dups by content hash; same file bytes won’t be re-saved/extracted. Old manual copies can be removed from data/uploads/.
	•	Chroma error like no such table: databases
The app auto-repairs. If needed, manually clear and reindex:

rm -rf data/processed/chroma


