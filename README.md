# AI Research Assistant — RAG System

Enterprise-grade RAG with hybrid BM25 + FAISS retrieval, cross-encoder reranking, LangGraph orchestration, and cited answers via FastAPI.

## Setup (2 steps)

### 1. Add your API key
```bash
cp .env.example .env
# Open .env and set: GEMINI_API_KEY=your-key-here
```
Get a free Gemini key at: https://aistudio.google.com/app/apikey

### 2. Run with Docker
```bash
docker-compose up --build
```

App: http://localhost:8000  
Swagger UI: http://localhost:8000/docs

---

## API

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | `/api/v1/documents/upload` | Upload a PDF (multipart/form-data, field: `file`) |
| GET  | `/api/v1/documents/` | List all uploaded documents |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| POST | `/api/v1/chat/` | Ask a question |
| GET  | `/api/v1/chat/{session_id}/history` | Get conversation history |
| DELETE | `/api/v1/chat/{session_id}` | Delete a session |
| GET  | `/health` | Health check |

### Upload a PDF
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@your_paper.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'
```

### Continue a conversation
```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-uuid", "query": "Tell me more about section 3"}'
```

---

## Architecture

```
PDF Upload
  → PyMuPDF (text extraction per page)
  → RecursiveCharacterTextSplitter (800 chars, 150 overlap)
  → Gemini text-embedding-004 (768-dim vectors, L2-normalised)
  → FAISS IndexIDMap+IndexFlatIP (vector store)
  → BM25Okapi corpus (keyword store)
  → PostgreSQL (Document + Chunk rows)

Query
  → LangGraph pipeline:
      1. Query Rewriter  — Gemini Flash expands + clarifies using chat history
      2. Hybrid Retriever — FAISS top-20 + BM25 top-20 fused with RRF (k=60)
      3. Cross-Encoder Reranker — ms-marco-MiniLM-L-6-v2 → top-5
      4. Answer Generator — Gemini Pro with grounded context + citation format
  → FastAPI JSON response: {answer, citations, rewritten_query, session_id}
```

## Rebuild index after deletions
```bash
docker-compose exec app python scripts/rebuild_index.py
```

## Resume bullet
> Built an enterprise-grade RAG assistant processing multi-PDF corpora with hybrid BM25+FAISS retrieval fused via Reciprocal Rank Fusion, cross-encoder reranking (ms-marco-MiniLM), and LangGraph orchestration; delivers cited answers via FastAPI with multi-session chat memory in PostgreSQL.
