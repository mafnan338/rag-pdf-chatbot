# Copy this file to ".env" and fill in the values you need.

# Which LLM backend to use: "ollama" (local, default) or "huggingface" (hosted API)

LLM_PROVIDER= groq

# --- Ollama settings (used when LLM_PROVIDER=ollama) ---

# Ollama must be running locally: `ollama run llama3.2`

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# --- Hugging Face settings (used when LLM_PROVIDER=huggingface) ---

# Get a free token at https://huggingface.co/settings/tokens

# Note: gated models like Llama-3.2 require you to accept the license on the

# model's Hugging Face page (while logged in) before your token can call it.

HUGGINGFACEHUB_API_TOKEN=your_hf_token_here

# Embedding model (runs locally via sentence-transformers, no key needed)

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Where uploaded PDFs / vector stores are cached on disk

DATA_DIR=./data

# Chunking

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# How many chunks to retrieve per question

RETRIEVAL_K=4

---

title: Marginalia RAG PDF Chatbot
emoji: 📄
colorFrom: yellow
colorTo: teal
sdk: docker
app_port: 7860
pinned: false

---

# Marginalia — Chat with a Research Paper (RAG)

A small full-stack app that lets you upload a research paper (PDF) and:

1. **Chat with it** — ask questions, answered only from the paper's own text (RAG), with conversation memory for follow-ups.
2. **Generate a plain-English blog post** from it.
3. **Auto-generate a glossary** of its key terms.

**Stack:** FastAPI backend + a plain HTML/CSS/JS frontend, served together in one container. RAG is built with LangChain, `sentence-transformers` embeddings, and a local FAISS vector store.
