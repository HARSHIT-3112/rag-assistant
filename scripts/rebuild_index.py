"""
Rebuilds FAISS + BM25 from all chunks in postgres.
Run after deleting documents to keep the index consistent.

Usage:
    python scripts/rebuild_index.py
"""
import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models.models import Chunk

settings = get_settings()
INDEX_PATH = Path(settings.faiss_index_dir)


async def rebuild():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Chunk).join(Chunk.document).order_by(Chunk.faiss_id))).scalars().all()

    if not rows:
        print("No chunks found.")
        return

    print(f"Embedding {len(rows)} chunks...")
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document",
    )

    all_vecs = []
    for i in range(0, len(rows), 100):
        batch = [r.text for r in rows[i:i+100]]
        all_vecs.extend(embedder.embed_documents(batch))
        print(f"  {min(i+100, len(rows))}/{len(rows)}")

    arr = np.array(all_vecs, dtype="float32")
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(arr, np.array([r.faiss_id for r in rows], dtype="int64"))
    faiss.write_index(index, str(INDEX_PATH / "index.faiss"))

    corpus = [r.text.lower().split() for r in rows]
    ids = [str(r.id) for r in rows]
    with open(INDEX_PATH / "bm25.pkl", "wb") as f:
        pickle.dump({"corpus": corpus, "ids": ids}, f)

    print(f"✓ Rebuilt index with {len(rows)} chunks.")


if __name__ == "__main__":
    asyncio.run(rebuild())
