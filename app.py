import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google import genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ── API client ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# ── ChromaDB ──────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
COLLECTION  = "who_facts"
EMBED_MODEL = "all-MiniLM-L6-v2"
DOCS_DIR    = Path("docs")

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# get_or_create so a brand new project (no chroma_db/ yet) still boots
collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},
)

# ── Constants ─────────────────────────────────────────────────
TOP_K                = 3
SIMILARITY_THRESHOLD = 0.55
RATE_LIMIT_DELAY     = 4  # seconds between Gemini calls
MAX_FILE_SIZE_BYTES  = 2 * 1024 * 1024  # 2 MB per file

SYSTEM_PROMPT = """You are NURA, a healthcare information assistant.
You answer questions ONLY using the document excerpts provided in the user message.

Rules you must follow:
1. Base your answer solely on the provided excerpts. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer the question, say so clearly and politely. Do not guess.
3. Keep answers concise and factual.
4. Do not provide personal medical advice or diagnoses."""


# ── Document chunking helpers (mirrors notebook Cell 3) ────────

def parse_document(filename: str, raw_text: str) -> dict:
    """Split off an optional SOURCE: header line, return {filename, source_url, text}."""
    raw = raw_text.strip()
    lines = raw.splitlines()
    source_url = ""
    if lines and lines[0].startswith("SOURCE:"):
        source_url = lines[0].replace("SOURCE:", "").strip()
        raw = "\n".join(lines[2:]).strip()
    return {"filename": filename, "source_url": source_url, "text": raw}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind('.', start + chunk_size - 80, end)
            if boundary == -1:
                boundary = text.rfind('\n', start + chunk_size - 80, end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if len(c) > 30]


def add_document_to_index(doc: dict):
    """Chunk a single document and add it to the live ChromaDB collection."""
    chunks = chunk_text(doc["text"])
    if not chunks:
        return 0

    ids = [f"{doc['filename']}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": doc["filename"],
            "source_url": doc["source_url"],
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def remove_document_from_index(filename: str):
    """Delete every chunk belonging to a document from ChromaDB."""
    collection.delete(where={"filename": filename})


def safe_filename(raw_name: str) -> str:
    """Strip .txt, keep only the document stem, block path traversal."""
    stem = Path(raw_name).stem
    stem = re.sub(r'[^a-zA-Z0-9_\-]', '', stem)
    if not stem:
        raise ValueError("Invalid filename.")
    return stem


def list_documents() -> list[dict]:
    """List documents currently on disk in docs/, with chunk counts from Chroma."""
    docs = []
    if not DOCS_DIR.exists():
        return docs

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    chunk_counts: dict[str, int] = {}
    for m in all_meta:
        chunk_counts[m["filename"]] = chunk_counts.get(m["filename"], 0) + 1

    for path in sorted(DOCS_DIR.glob("*.txt")):
        stem = path.stem
        docs.append({
            "filename": stem,
            "chunks": chunk_counts.get(stem, 0),
            "size_bytes": path.stat().st_size,
        })
    return docs


# ── RAG pipeline ──────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "filename": meta["filename"],
            "source_url": meta.get("source_url", ""),
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
        })
    return chunks


def is_out_of_scope(chunks: list[dict]) -> bool:
    if not chunks:
        return True
    return chunks[0]["distance"] > SIMILARITY_THRESHOLD


def build_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Excerpt {i} — Source: {chunk['filename']}.txt]")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines).strip()


def current_doc_names() -> list[str]:
    return [d["filename"] for d in list_documents()]


def ask_nura(question: str) -> dict:
    chunks = retrieve(question)

    if is_out_of_scope(chunks):
        loaded = ", ".join(current_doc_names()) or "no documents currently loaded"
        return {
            "answer": (
                f"I'm sorry, I can only answer questions based on the documents "
                f"I have access to ({loaded}). Your question doesn't appear to be "
                f"covered by these documents."
            ),
            "sources": [],
            "out_of_scope": True,
        }

    context = build_context(chunks)
    user_message = f"""Answer the following question using ONLY the excerpts below.
If the excerpts don't fully answer it, say so.

Question: {question}

Excerpts:
{context}"""

    time.sleep(RATE_LIMIT_DELAY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=user_message,
        config={"system_instruction": SYSTEM_PROMPT, "max_output_tokens": 512},
    )

    seen = set()
    sources = []
    for c in chunks:
        if c["filename"] not in seen:
            seen.add(c["filename"])
            sources.append({
                "filename": c["filename"],
                "url": c["source_url"] or None,
            })

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "out_of_scope": False,
    }


# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(title="Mini NURA", description="WHO fact sheet RAG assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    filename: str
    url: str | None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    out_of_scope: bool


class DocumentItem(BaseModel):
    filename: str
    chunks: int
    size_bytes: int


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_loaded": current_doc_names(),
        "chunks": collection.count(),
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = ask_nura(req.question.strip())
    return AnswerResponse(**result)


@app.get("/documents", response_model=list[DocumentItem])
def get_documents():
    return list_documents()


@app.post("/documents", response_model=DocumentItem)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 2 MB).")
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        stem = safe_filename(file.filename)
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    DOCS_DIR.mkdir(exist_ok=True)
    dest_path = DOCS_DIR / f"{stem}.txt"

    is_replace = dest_path.exists()
    if is_replace:
        remove_document_from_index(stem)

    dest_path.write_text(raw_text, encoding="utf-8")

    doc = parse_document(stem, raw_text)
    n_chunks = add_document_to_index(doc)

    if n_chunks == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Document produced no usable content after chunking (too short).",
        )

    return DocumentItem(filename=stem, chunks=n_chunks, size_bytes=len(raw_bytes))


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    stem = safe_filename(filename)
    path = DOCS_DIR / f"{stem}.txt"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    remove_document_from_index(stem)
    path.unlink()

    return {"status": "deleted", "filename": stem}


# ── Static files (HTML frontend) ──────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
