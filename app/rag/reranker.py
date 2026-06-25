from __future__ import annotations
from functools import lru_cache
from sentence_transformers import CrossEncoder
from app.core.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    # Loaded once, stays in memory. Fast cross-encoder — sees (query, chunk) jointly.
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    k = top_k or settings.top_k_rerank
    if not candidates:
        return []

    scores = _get_model().predict(
        [(query, c["text"]) for c in candidates],
        show_progress_bar=False,
    )
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:k]
