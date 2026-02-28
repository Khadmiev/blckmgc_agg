from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.dependencies import get_current_user, get_db
from app.models.message import Message
from app.models.thread import Thread
from app.models.user import User
from app.schemas.message import (
    MediaAttachmentResponse,
    MessageResponse,
    PaginatedMessages,
    ThreadDetailResponse,
)
from app.schemas.thread import ThreadCreate, ThreadListItem, ThreadResponse, ThreadUpdate
from app.services.thread_service import generate_thread_title

router = APIRouter()


async def _get_user_thread(
    thread_id: UUID, user: User, db: AsyncSession
) -> Thread:
    result = await db.execute(
        select(Thread).where(
            Thread.id == thread_id,
            Thread.user_id == user.id,
            Thread.is_deleted == False,  # noqa: E712
        )
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return thread


@router.get("/", response_model=list[ThreadListItem])
async def list_threads(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    latest_msg = (
        select(Message.content)
        .where(
            Message.thread_id == Thread.id,
            Message.role.in_(["user", "assistant"]),
        )
        .order_by(Message.created_at.desc())
        .limit(1)
        .correlate(Thread)
        .scalar_subquery()
    )
    last_msg_preview = func.substring(latest_msg, 1, 100).label("preview")

    stmt = (
        select(Thread, last_msg_preview)
        .where(Thread.user_id == user.id, Thread.is_deleted == False)  # noqa: E712
        .order_by(Thread.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [
        ThreadListItem(
            id=thread.id,
            title=thread.title,
            llm_name=thread.llm_name,
            updated_at=thread.updated_at,
            last_message_preview=preview,
        )
        for thread, preview in rows
    ]


@router.post("/", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: ThreadCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = Thread(
        user_id=user.id,
        title=body.title,
        llm_name=body.llm_name,
    )
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    return thread


@router.get("/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(
    thread_id: UUID,
    msg_offset: int = Query(0, ge=0),
    msg_limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_user_thread(thread_id, user, db)

    msg_stmt = (
        select(Message)
        .where(Message.thread_id == thread.id)
        .order_by(Message.created_at)
        .offset(msg_offset)
        .limit(msg_limit)
        .options(selectinload(Message.attachments))
    )
    msg_result = await db.execute(msg_stmt)
    messages = msg_result.scalars().all()

    return ThreadDetailResponse(
        id=thread.id,
        title=thread.title,
        llm_name=thread.llm_name,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                model=m.model,
                token_count=m.token_count,
                created_at=m.created_at,
                attachments=[
                    MediaAttachmentResponse(
                        id=a.id,
                        media_type=a.media_type,
                        mime_type=a.mime_type,
                        file_size=a.file_size,
                        has_thumbnail=a.thumbnail_path is not None,
                    )
                    for a in m.attachments
                ],
            )
            for m in messages
        ],
    )


@router.post("/{thread_id}/generate-title")
async def generate_title(
    thread_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_user_thread(thread_id, user, db)
    title = await generate_thread_title(db, thread)
    return {"title": title}


@router.patch("/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: UUID,
    body: ThreadUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_user_thread(thread_id, user, db)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(thread, field, value)
    await db.commit()
    await db.refresh(thread)
    return thread


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_user_thread(thread_id, user, db)
    thread.is_deleted = True
    await db.commit()
