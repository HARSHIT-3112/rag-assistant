from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    total_pages: int
    total_chunks: int
    created_at: datetime
    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    session_id: UUID | None = Field(default=None)
    query: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    index: int
    filename: str
    page_number: int
    chunk_id: UUID
    document_id: UUID
    excerpt: str


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    citations: list[CitationOut]
    rewritten_query: str


class ChatMessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    model_config = {"from_attributes": True}


class ChatHistoryOut(BaseModel):
    session_id: UUID
    messages: list[ChatMessageOut]
