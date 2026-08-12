"""
rag_utils.py
------------
Document loading -> cleaning -> chunking -> embedding -> vector store.

Uses:
  - pypdf (via LangChain's PyPDFLoader) for PDF text extraction
  - RecursiveCharacterTextSplitter for chunking
  - FAISS as the local vector database
"""

import os
import re
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from llm_utils import get_embeddings

DATA_DIR = os.getenv("DATA_DIR", "./data")


def _session_dir(session_id: str) -> str:
    path = os.path.join(DATA_DIR, session_id)
    os.makedirs(path, exist_ok=True)
    return path


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, drop page-number-only lines,
    strip common PDF extraction artifacts."""
    text = text.replace("\x0c", " ")            # form-feed from page breaks
    text = re.sub(r"-\n(?=[a-z])", "", text)     # de-hyphenate line-wrapped words
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"\d{1,4}", ln)]
    return "\n".join(lines).strip()


def load_and_chunk_pdf(pdf_path: str) -> list:
    """Load a PDF, clean each page's text, and split into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # one LangChain Document per page

    for page in pages:
        page.page_content = clean_text(page.page_content)

    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_vectorstore(session_id: str, chunks: list) -> FAISS:
    """Embed chunks and persist a FAISS index to disk for this session."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(os.path.join(_session_dir(session_id), "faiss_index"))
    return vectorstore


def load_vectorstore(session_id: str) -> FAISS:
    embeddings = get_embeddings()
    index_path = os.path.join(_session_dir(session_id), "faiss_index")
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


def retrieve_chunks(session_id: str, query: str, k: int = 4) -> list:
    vectorstore = load_vectorstore(session_id)
    return vectorstore.similarity_search(query, k=k)


def full_text_from_chunks(chunks: list, max_chars: int = 12000) -> str:
    """Concatenate chunk text (deduplicated by page order) up to a char budget,
    used as source material for blog/glossary generation."""
    text = "\n\n".join(c.page_content for c in chunks)
    return text[:max_chars]


def delete_session(session_id: str) -> None:
    path = os.path.join(DATA_DIR, session_id)
    if os.path.isdir(path):
        shutil.rmtree(path)
