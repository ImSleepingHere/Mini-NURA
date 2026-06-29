import os
import time
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
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

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION, embedding_function=embed_fn)

# ── Constants ─────────────────────────────────────────────────
TOP_K               = 3
SIMILARITY_THRESHOLD = 0.55
RATE_LIMIT_DELAY    = 4  # seconds between Gemini calls

SYSTEM_PROMPT = """You are NURA, a healthcare information assistant.
You answer questions ONLY using the document excerpts provided in the user message.

Rules you must follow:
1. Base your answer solely on the provided excerpts. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer the question, say so clearly and politely. Do not guess.
3. Keep answers concise and factual.
4. Do not provide personal medical advice or diagnoses."""

DOCS = ["Asthma", "Botulism", "Hantavirus", "Tetanus", "YellowFever"]

# ── RAG pipeline ──────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
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


def ask_nura(question: str) -> dict:
    chunks = retrieve(question)

    if is_out_of_scope(chunks):
        return {
            "answer": (
                "I'm sorry, I can only answer questions based on the WHO fact sheets "
                "I have access to (Asthma, Botulism, Hantavirus, Tetanus, Yellow Fever). "
                "Your question doesn't appear to be covered by these documents."
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
        model="gemini-2.5-flash",
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


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "docs_loaded": DOCS, "chunks": collection.count()}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = ask_nura(req.question.strip())
    return AnswerResponse(**result)


# ── Static files (HTML frontend) ──────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
