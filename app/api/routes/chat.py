import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.models import ChatMessage, ChatSession
from app.rag.graph import run_rag
from app.schemas.schemas import ChatHistoryOut, ChatMessageOut, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


async def _get_or_create_session(session_id: uuid.UUID | None, db: AsyncSession) -> ChatSession:
    if session_id:
        result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(404, "Session not found.")
        return session
    session = ChatSession()
    db.add(session)
    await db.flush()
    return session


async def _load_history(session: ChatSession, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    )
    return [{"role": m.role, "content": m.content} for m in result.scalars().all()]


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_or_create_session(request.session_id, db)
    history = await _load_history(session, db)

    try:
        result = await run_rag(query=request.query, chat_history=history, db=db)
    except Exception as e:
        raise HTTPException(500, f"RAG pipeline error: {e}")

    db.add_all([
        ChatMessage(session_id=session.id, role="user", content=request.query),
        ChatMessage(session_id=session.id, role="assistant", content=result["answer"]),
    ])
    await db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=result["answer"],
        citations=result["citations"],
        rewritten_query=result["rewritten_query"],
    )


@router.get("/{session_id}/history", response_model=ChatHistoryOut)
async def get_history(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    if not result.scalar_one_or_none():
        raise HTTPException(404, "Session not found.")

    msgs = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    return ChatHistoryOut(
        session_id=session_id,
        messages=[ChatMessageOut.model_validate(m) for m in msgs.scalars().all()],
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found.")
    await db.delete(session)
    await db.commit()
