from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MediaAttachmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    media_type: str
    mime_type: str
    file_size: int
    has_thumbnail: bool = False


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: Optional[str]
    model: Optional[str]
    token_count: Optional[int]
    created_at: datetime
    attachments: list[MediaAttachmentResponse] = []


class ThreadDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    llm_name: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []


class PaginatedMessages(BaseModel):
    items: list[MessageResponse]
    total: int
    offset: int
    limit: int
