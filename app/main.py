from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents
from app.core.config import get_settings
from app.db.session import Base, engine
from app.ingestion.pipeline import get_faiss

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables + warm FAISS into memory
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    get_faiss()
    print("✓ Database tables ready")
    print("✓ FAISS index loaded")
    yield
    await engine.dispose()


app = FastAPI(
    title="AI Research Assistant",
    description="Enterprise RAG with hybrid search, reranking, and cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
