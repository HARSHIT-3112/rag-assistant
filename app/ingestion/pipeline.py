import io
import pickle
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.models import Chunk, Document

settings = get_settings()

# ── In-memory index state (loaded once at startup, updated on each ingest) ───
_faiss_index = None
_bm25_corpus: list[list[str]] = []
_bm25_chunk_ids: list[str] = []
_bm25_index: BM25Okapi | None = None
_next_faiss_id: int = 0

INDEX_PATH = Path(settings.faiss_index_dir)


def _load_or_create_faiss():
    global _faiss_index, _next_faiss_id
    import faiss

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    idx_file = INDEX_PATH / "index.faiss"

    if idx_file.exists():
        _faiss_index = faiss.read_index(str(idx_file))
        _next_faiss_id = _faiss_index.ntotal
    else:
        # 768 dims = Gemini embedding-001 output size
        # IndexFlatIP = inner product (cosine when vectors are L2-normalised)
        # IndexIDMap = lets us assign our own integer IDs instead of 0,1,2...
        _faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        _next_faiss_id = 0


def _persist():
    import faiss
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_faiss_index, str(INDEX_PATH / "index.faiss"))
    with open(INDEX_PATH / "bm25.pkl", "wb") as f:
        pickle.dump({"corpus": _bm25_corpus, "ids": _bm25_chunk_ids}, f)


def _rebuild_bm25_index():
    global _bm25_index
    if _bm25_corpus:
        _bm25_index = BM25Okapi(_bm25_corpus)
    else:
        _bm25_index = None


def _load_bm25():
    global _bm25_corpus, _bm25_chunk_ids
    p = INDEX_PATH / "bm25.pkl"
    if p.exists():
        with open(p, "rb") as f:
            data = pickle.load(f)
            _bm25_corpus = data["corpus"]
            _bm25_chunk_ids = data["ids"]
    _rebuild_bm25_index()


def get_faiss():
    global _faiss_index
    if _faiss_index is None:
        _load_or_create_faiss()
        _load_bm25()
    return _faiss_index


def get_bm25():
    get_faiss()
    return _bm25_index


# ── Step 1: Extract text per page ────────────────────────────────────────────
def extract_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    idx = 0
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({"chunk_index": idx, "page_number": page["page"], "text": piece})
            idx += 1
    return chunks


# ── Step 3: Embed ─────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.gemini_api_key,
    )
    vectors = embedder.embed_documents(texts)
    arr = np.array(vectors, dtype="float32")
    # L2 normalise so inner product == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-9)


# ── Step 4: Index into FAISS + BM25 ──────────────────────────────────────────
def index_chunks(chunks: list[dict], embeddings: np.ndarray, chunk_uuids: list[str]) -> list[int]:
    global _next_faiss_id

    index = get_faiss()
    faiss_ids = list(range(_next_faiss_id, _next_faiss_id + len(chunks)))
    index.add_with_ids(embeddings, np.array(faiss_ids, dtype="int64"))
    _next_faiss_id += len(chunks)

    for i, chunk in enumerate(chunks):
        _bm25_corpus.append(chunk["text"].lower().split())
        _bm25_chunk_ids.append(chunk_uuids[i])

    _rebuild_bm25_index()
    _persist()
    return faiss_ids


# ── Remove chunks from indexes (called on document delete) ───────────────────
async def remove_document_chunks(doc_id: str, db: AsyncSession):
    from sqlalchemy import select as sa_select
    result = await db.execute(
        sa_select(Chunk.faiss_id, Chunk.id).where(Chunk.document_id == doc_id)
    )
    rows = result.all()
    if not rows:
        return

    faiss_ids_to_remove = [r[0] for r in rows]
    chunk_uuids_to_remove = {str(r[1]) for r in rows}

    index = get_faiss()
    index.remove_ids(np.array(faiss_ids_to_remove, dtype="int64"))

    global _bm25_corpus, _bm25_chunk_ids
    paired = [
        (corpus, cid)
        for corpus, cid in zip(_bm25_corpus, _bm25_chunk_ids)
        if cid not in chunk_uuids_to_remove
    ]
    if paired:
        _bm25_corpus, _bm25_chunk_ids = [list(t) for t in zip(*paired)]
    else:
        _bm25_corpus = []
        _bm25_chunk_ids = []

    _rebuild_bm25_index()
    _persist()


# ── Step 5: Full pipeline (public entry point) ────────────────────────────────
async def ingest_pdf(pdf_bytes: bytes, filename: str, db: AsyncSession) -> Document:
    pages = extract_pages(pdf_bytes)
    if not pages:
        raise ValueError("Could not extract any text from this PDF.")

    chunks = chunk_pages(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    chunk_uuids = [str(uuid.uuid4()) for _ in chunks]
    faiss_ids = index_chunks(chunks, embeddings, chunk_uuids)

    doc = Document(
        filename=filename,
        total_pages=max(c["page_number"] for c in chunks),
        total_chunks=len(chunks),
    )
    db.add(doc)
    await db.flush()

    db.add_all([
        Chunk(
            id=uuid.UUID(chunk_uuids[i]),
            document_id=doc.id,
            chunk_index=c["chunk_index"],
            page_number=c["page_number"],
            text=c["text"],
            faiss_id=faiss_ids[i],
        )
        for i, c in enumerate(chunks)
    ])

    await db.commit()
    await db.refresh(doc)
    return doc
