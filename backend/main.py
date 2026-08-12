"""
main.py
-------
FastAPI backend for the "Chat with a Research Paper" RAG app.

Endpoints:
  POST   /api/upload            -> upload + parse + index a PDF, returns session_id
  POST   /api/blog               -> generate a plain-English blog post
  POST   /api/glossary           -> generate a structured glossary (JSON)
  POST   /api/chat               -> ask a question, answered via RAG + memory
  GET    /api/session/{id}       -> session status (filename, #chunks, history)
  DELETE /api/session/{id}       -> clear a session's data

Run with:
  uvicorn main:app --reload --port 8000
"""

import json
import os
import re
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_utils import (
    load_and_chunk_pdf,
    build_vectorstore,
    full_text_from_chunks,
    delete_session,
)
from llm_utils import get_llm, BLOG_PROMPT, GLOSSARY_PROMPT, RAG_CHAT_PROMPT

UPLOAD_DIR = os.path.join(os.getenv("DATA_DIR", "./data"), "_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Chat-with-a-Paper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store.
#   { session_id: {
#         "filename": str,
#         "vectorstore": FAISS,
#         "chunks": [Document, ...],
#         "history": [(question, answer), ...],
#     }
#   }
# For a production app, swap this for Redis / a database.
# ---------------------------------------------------------------------------
SESSIONS: dict = {}


def _get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Unknown session_id. Upload a PDF first via /api/upload.",
        )
    return session


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SessionRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# 1. Upload a research paper
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        chunks = load_and_chunk_pdf(save_path)
        if not chunks:
            raise ValueError("No extractable text found in this PDF.")
        vectorstore = build_vectorstore(session_id, chunks)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {exc}")

    SESSIONS[session_id] = {
        "filename": file.filename,
        "vectorstore": vectorstore,
        "chunks": chunks,
        "history": [],
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "num_pages": max((c.metadata.get("page", 0) for c in chunks), default=0) + 1,
        "num_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# 2. Generate a plain-English blog
# ---------------------------------------------------------------------------

@app.post("/api/blog")
async def generate_blog(req: SessionRequest):
    session = _get_session(req.session_id)
    paper_text = full_text_from_chunks(session["chunks"])

    llm = get_llm()
    chain = BLOG_PROMPT | llm
    result = chain.invoke({"paper_text": paper_text})

    return {"blog_markdown": result.content}


# ---------------------------------------------------------------------------
# 3. Auto-generated glossary
# ---------------------------------------------------------------------------

@app.post("/api/glossary")
async def generate_glossary(req: SessionRequest):
    session = _get_session(req.session_id)
    paper_text = full_text_from_chunks(session["chunks"])

    llm = get_llm()
    chain = GLOSSARY_PROMPT | llm
    result = chain.invoke({"paper_text": paper_text})

    raw = result.content.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        glossary = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not match:
            raise HTTPException(
                status_code=502, detail="Model did not return valid JSON."
            )
        glossary = json.loads(match.group(0))

    return {"glossary": glossary}


# ---------------------------------------------------------------------------
# 4 & 5. Chat with the paper (RAG) + conversation memory
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = _get_session(req.session_id)
    vectorstore = session["vectorstore"]
    k = int(os.getenv("RETRIEVAL_K", 4))

    # Retrieval: pull the chunks most relevant to the *current* question
    docs = vectorstore.similarity_search(req.message, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Memory: fold prior turns into the prompt so follow-ups like
    # "compare it with the previous answer" resolve correctly
    history_text = "\n".join(
        f"User: {q}\nAssistant: {a}" for q, a in session["history"][-6:]
    ) or "(no previous turns)"

    llm = get_llm()
    chain = RAG_CHAT_PROMPT | llm
    result = chain.invoke(
        {
            "chat_history": history_text,
            "context": context,
            "question": req.message,
        }
    )
    answer = result.content

    session["history"].append((req.message, answer))

    sources = [
        {
            "page": d.metadata.get("page", None),
            "excerpt": d.page_content[:220].strip() + "...",
        }
        for d in docs
    ]

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.get("/api/session/{session_id}")
async def session_status(session_id: str):
    session = _get_session(session_id)
    return {
        "filename": session["filename"],
        "num_chunks": len(session["chunks"]),
        "turns": len(session["history"]),
        "history": session["history"],
    }


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    SESSIONS.pop(session_id, None)
    delete_session(session_id)
    return {"status": "deleted"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
