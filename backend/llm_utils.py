"""
llm_utils.py
------------
Centralizes:
  - LLM provider selection (Ollama running Llama 3.2 locally, or Hugging Face's
    hosted Inference API)
  - Embedding model loading
  - Prompt templates for: Blog generation, Glossary extraction, and RAG chat

Swapping models is just an .env change (LLM_PROVIDER=ollama|huggingface).
"""

import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings


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
            # Preferred: the dedicated langchain-ollama package
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fallback if only langchain-community is installed
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
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_name=model_name,
    )

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
            "You are a research assistant answering questions about ONE specific "
            "paper. Answer strictly using the provided context chunks retrieved "
            "from that paper. If the answer is not contained in the context, say "
            "you couldn't find that in the paper -- do not make anything up. "
            "Keep answers concise and cite the relevant idea in your own words. "
            "You may use the prior conversation to resolve references like "
            "'it', 'that figure', or 'compare it with the previous answer'.",
        ),
        (
            "human",
            "Conversation so far:\n{chat_history}\n\n"
            "Retrieved context from the paper:\n{context}\n\n"
            "Question: {question}",
        ),
    ]
)