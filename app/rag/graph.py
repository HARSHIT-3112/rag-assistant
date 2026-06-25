from __future__ import annotations
from typing import TypedDict, Any

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.rag.reranker import rerank
from app.rag.retriever import hybrid_retrieve

settings = get_settings()


class RAGState(TypedDict):
    original_query: str
    chat_history: list[dict]
    db: Any                    # AsyncSession — Any avoids TypedDict serialisation issues
    rewritten_query: str
    query_embedding: list[float]
    candidates: list[dict]
    reranked: list[dict]
    answer: str
    citations: list[dict]


# ── Node 1: Query rewriting ───────────────────────────────────────────────────
async def rewrite_query(state: RAGState) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )
    history = "".join(
        f"{m['role'].capitalize()}: {m['content']}\n"
        for m in state["chat_history"][-6:]
    )
    prompt = (
        "You are a query optimizer for a document retrieval system.\n"
        "Rewrite the user's question to be self-contained (resolve pronouns using history), "
        "expanded with synonyms, and specific. Output ONLY the rewritten query.\n\n"
        f"History:\n{history}\n"
        f"Question: {state['original_query']}"
    )
    resp = await llm.ainvoke(prompt)
    return {"rewritten_query": resp.content.strip()}


# ── Node 2: Embed + Hybrid retrieve ──────────────────────────────────────────
async def retrieve(state: RAGState) -> dict:
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.gemini_api_key,
    )
    vec = await embedder.aembed_query(state["rewritten_query"])
    arr = np.array(vec, dtype="float32")
    arr = arr / max(float(np.linalg.norm(arr)), 1e-9)

    candidates = await hybrid_retrieve(
        query_embedding=arr,
        query_text=state["rewritten_query"],
        db=state["db"],
    )
    return {"query_embedding": vec, "candidates": candidates}


# ── Node 3: Rerank ────────────────────────────────────────────────────────────
def rerank_node(state: RAGState) -> dict:
    return {"reranked": rerank(state["rewritten_query"], state["candidates"])}


# ── Node 4: Generate answer with citations ────────────────────────────────────
async def generate_answer(state: RAGState) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )

    if not state["reranked"]:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents to answer your question.",
            "citations": [],
        }

    context = "\n\n---\n\n".join(
        f"[{i+1}] (File: {c['filename']}, Page {c['page_number']})\n{c['text']}"
        for i, c in enumerate(state["reranked"])
    )
    history = "".join(
        f"{m['role'].capitalize()}: {m['content']}\n"
        for m in state["chat_history"][-6:]
    )
    prompt = (
        "You are a precise research assistant. Answer using ONLY the provided context.\n"
        "Cite sources inline with [1], [2], etc. "
        "If the context is insufficient, say so explicitly. Use markdown.\n\n"
        f"Context:\n{context}\n\n"
        f"History:\n{history}\n"
        f"Question: {state['original_query']}\n\nAnswer:"
    )

    resp = await llm.ainvoke(prompt)
    citations = [
        {
            "index": i + 1,
            "filename": c["filename"],
            "page_number": c["page_number"],
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "excerpt": c["text"][:200] + "...",
        }
        for i, c in enumerate(state["reranked"])
    ]
    return {"answer": resp.content.strip(), "citations": citations}


# ── Build + compile graph (module-level singleton) ────────────────────────────
def _build():
    g = StateGraph(RAGState)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank_node)
    g.add_node("generate_answer", generate_answer)
    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate_answer")
    g.add_edge("generate_answer", END)
    return g.compile()


rag_graph = _build()


async def run_rag(query: str, chat_history: list[dict], db: AsyncSession) -> dict:
    state = await rag_graph.ainvoke({
        "original_query": query,
        "chat_history": chat_history,
        "db": db,
        "rewritten_query": "",
        "query_embedding": [],
        "candidates": [],
        "reranked": [],
        "answer": "",
        "citations": [],
    })
    return {
        "answer": state["answer"],
        "citations": state["citations"],
        "rewritten_query": state["rewritten_query"],
    }
