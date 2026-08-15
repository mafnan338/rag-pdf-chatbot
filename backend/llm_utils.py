"""
llm_utils.py
------------
Centralizes:
  - LLM provider selection: Ollama (local Llama 3.2), Groq (free hosted API,
    generous limits), or Hugging Face's hosted Inference API
  - Embeddings via Hugging Face's router API (no local model load — keeps
    memory usage low enough for free-tier hosting like Render)
  - Prompt templates for: Blog generation, Glossary extraction, and RAG chat

Swapping models is just an .env change (LLM_PROVIDER=ollama|groq|huggingface).
"""

import os
from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate


class HFRouterEmbeddings(Embeddings):
    """
    Calls Hugging Face's *current* router endpoint for feature-extraction.
    langchain_community's HuggingFaceInferenceAPIEmbeddings still targets the
    legacy api-inference.huggingface.co host, which Hugging Face has fully
    decommissioned (returns DNS failures / 410 Gone) — this talks to the
    replacement (router.huggingface.co) directly instead.
    """

    def __init__(self, model_name: str, api_key: str, batch_size: int = 32):
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model_name}/pipeline/feature-extraction"
        )

    def _embed_batch(self, texts):
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"inputs": texts},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return vectors

    def embed_query(self, text):
        return self._embed_batch([text])[0]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm():
    """Return a chat-model instance based on LLM_PROVIDER in the environment.
    Defaults to Ollama running Llama 3.2 locally (no API key required)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=0.3,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

    # huggingface (used only if LLM_PROVIDER=huggingface)
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    endpoint = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.3,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Embeddings via Hugging Face's hosted router API — not loaded locally.
    This avoids pulling torch + the model weights into the app's own process,
    which is what was pushing memory past free-tier hosting limits (e.g.
    Render's 512MB free web services). Requires HUGGINGFACEHUB_API_TOKEN to
    be set, even if your chat LLM provider is Groq/Ollama/etc — embeddings
    are independent of which provider answers chat questions.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Embeddings require a Hugging "
            "Face token even when using Groq/Ollama as the chat LLM provider."
        )
    return HFRouterEmbeddings(model_name=model_name, api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BLOG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a science communicator who turns dense research papers into "
            "friendly, plain-English blog posts for curious non-experts. "
            "Use short sentences, concrete analogies, and a warm, engaging tone. "
            "Avoid jargon; when a technical term is unavoidable, explain it in "
            "parentheses the first time it appears. Structure the output with a "
            "catchy title, a short hook paragraph, 3-5 sections with subheadings "
            "covering: the problem, the idea/approach, how it works (with an "
            "analogy), and why it matters. Keep the whole post under 600 words. "
            "Output in Markdown.",
        ),
        (
            "human",
            "Here are excerpts from the research paper (may be truncated):\n\n"
            "{paper_text}\n\n"
            "Write the plain-English blog post now.",
        ),
    ]
)

GLOSSARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract technical terms from research paper text and explain each "
            "one in one simple sentence a beginner could understand. "
            "Return ONLY a valid JSON array, nothing else -- no markdown fences, "
            "no preamble. Each element must look like: "
            '{{"term": "...", "meaning": "..."}}. '
            "Pick the 8-15 most important technical terms actually used in the "
            "text. Do not invent terms that are not present.",
        ),
        (
            "human",
            "Paper excerpts:\n\n{paper_text}\n\nReturn the JSON glossary now.",
        ),
    ]
)

RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant answering questions about a "
            "specific paper. Answer ONLY using the provided context excerpts from "
            "the paper. If the answer isn't in the context, say you can't find "
            "that in the paper rather than guessing or using outside knowledge. "
            "Use the conversation history to resolve follow-up questions (e.g. "
            "'it', 'that', 'compare it with the previous answer'). Be concise "
            "and cite specific details from the context where relevant.",
        ),
        (
            "human",
            "Conversation so far:\n{chat_history}\n\n"
            "Context excerpts from the paper:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer using only the context above.",
        ),
    ]
)