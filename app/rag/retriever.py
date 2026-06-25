from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ingestion.pipeline import _bm25_chunk_ids, get_bm25, get_faiss
from app.models.models import Chunk, Document

settings = get_settings()
RRF_K = 60  # standard Reciprocal Rank Fusion constant


async def hybrid_retrieve(
    query_embedding: np.ndarray,
    query_text: str,
    db: AsyncSession,
    top_k: int | None = None,
) -> list[dict]:
    top_k_final = top_k or (settings.top_k_rerank * 3)

    # ── Dense: FAISS ─────────────────────────────────────────────────────────
    index = get_faiss()
    q = query_embedding.reshape(1, -1).astype("float32")
    distances, faiss_ids = index.search(q, settings.top_k_vector)
    faiss_ranks: dict[int, int] = {
        int(fid): rank for rank, fid in enumerate(faiss_ids[0]) if fid != -1
    }

    # ── Sparse: BM25 ─────────────────────────────────────────────────────────
    bm25_ranks: dict[str, int] = {}
    bm25 = get_bm25()
    if bm25 is not None:
        scores = bm25.get_scores(query_text.lower().split())
        top_idx = np.argsort(scores)[::-1][:settings.top_k_bm25]
        for rank, idx in enumerate(top_idx):
            if scores[idx] > 0:
                bm25_ranks[_bm25_chunk_ids[idx]] = rank

    # ── Fetch metadata from DB ────────────────────────────────────────────────
    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Chunk.faiss_id.in_(list(faiss_ranks.keys())) |
            Chunk.id.in_(list(bm25_ranks.keys()))
        )
    )
    rows = (await db.execute(stmt)).all()

    faiss_id_to_meta: dict[int, dict] = {}
    uuid_to_meta: dict[str, dict] = {}
    for chunk, doc in rows:
        meta = {
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "text": chunk.text,
            "page_number": chunk.page_number,
            "filename": doc.filename,
        }
        faiss_id_to_meta[chunk.faiss_id] = meta
        uuid_to_meta[str(chunk.id)] = meta

    # ── RRF fusion ────────────────────────────────────────────────────────────
    # score = Σ 1/(k + rank)  — higher is better
    rrf: dict[str, float] = {}

    for fid, rank in faiss_ranks.items():
        meta = faiss_id_to_meta.get(fid)
        if meta:
            cid = meta["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    for cid, rank in bm25_ranks.items():
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k_final]
    return [{**uuid_to_meta[cid], "score": rrf[cid]} for cid in sorted_ids if cid in uuid_to_meta]
