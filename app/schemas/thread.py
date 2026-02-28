from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ThreadCreate(BaseModel):
    title: str = "New chat"
    llm_name: str


class ThreadUpdate(BaseModel):
    title: Optional[str] = None
    llm_name: Optional[str] = None


class ThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    llm_name: str
    created_at: datetime
    updated_at: datetime


class ThreadListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    llm_name: str
    updated_at: datetime
    last_message_preview: Optional[str] = None
