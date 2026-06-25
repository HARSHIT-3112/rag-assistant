from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.ingestion.pipeline import ingest_pdf, remove_document_chunks
from app.models.models import Document
from app.schemas.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])
MAX_MB = 50


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(file: UploadFile, db: AsyncSession = Depends(get_db)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(415, "Only PDF files are supported.")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB} MB.")

    try:
        doc = await ingest_pdf(pdf_bytes=content, filename=file.filename or "upload.pdf", db=db)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")

    return doc


@router.get("/", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found.")
    await remove_document_chunks(str(doc.id), db)
    await db.delete(doc)
    await db.commit()
