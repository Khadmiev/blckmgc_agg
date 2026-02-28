from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_current_user, get_db
from app.models.message import Message
from app.models.thread import Thread
from app.models.user import User
from app.services.chat_service import (
    process_uploaded_files,
    save_user_message,
    stream_llm_response,
)
from app.storage.base import get_storage_backend

router = APIRouter()


async def _get_user_thread(thread_id: UUID, user: User, db: AsyncSession) -> Thread:
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


@router.post("/{thread_id}/send")
async def send_message(
    thread_id: UUID,
    prompt: str = Form(...),
    files: Optional[list[UploadFile]] = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_user_thread(thread_id, user, db)
    storage = get_storage_backend()

    attachments = []
    user_content: str | list[dict] = prompt

    if files:
        files_data = []
        for f in files:
            data = await f.read()
            files_data.append((f.filename or "upload", data, f.content_type or "application/octet-stream"))
        attachments = await process_uploaded_files(files_data, storage, thread.id)

        if any(a.media_type == "image" for a in attachments):
            content_parts: list[dict] = [{"type": "text", "text": prompt}]
            for att in attachments:
                if att.media_type == "image":
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"attachment://{att.id}"},
                    })
            user_content = content_parts

    user_msg = await save_user_message(db, thread, prompt, attachments)

    async def event_generator():
        async for event in stream_llm_response(db, thread, user_content, storage, attachments=attachments):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]) if isinstance(event["data"], dict) else event["data"],
            }

    return EventSourceResponse(event_generator())


@router.post("/{thread_id}/regenerate")
async def regenerate_response(
    thread_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    thread = await _get_user_thread(thread_id, user, db)
    storage = get_storage_backend()

    last_user_msg_result = await db.execute(
        select(Message)
        .where(Message.thread_id == thread.id, Message.role == "user")
        .order_by(Message.created_at.desc())
        .limit(1)
        .options(selectinload(Message.attachments))
    )
    last_user_msg = last_user_msg_result.scalar_one_or_none()
    if last_user_msg is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No user message to regenerate from")

    last_assistant_result = await db.execute(
        select(Message)
        .where(
            Message.thread_id == thread.id,
            Message.role == "assistant",
            Message.created_at > last_user_msg.created_at,
        )
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    last_assistant_msg = last_assistant_result.scalar_one_or_none()
    if last_assistant_msg:
        await db.delete(last_assistant_msg)
        await db.commit()

    user_content: str | list[dict] = last_user_msg.content or ""
    if last_user_msg.attachments:
        content_parts: list[dict] = []
        if last_user_msg.content:
            content_parts.append({"type": "text", "text": last_user_msg.content})
        for att in last_user_msg.attachments:
            if att.media_type == "image":
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"attachment://{att.id}"},
                })
        if content_parts:
            user_content = content_parts

    regen_attachments = list(last_user_msg.attachments) if last_user_msg.attachments else None

    async def event_generator():
        async for event in stream_llm_response(db, thread, user_content, storage, attachments=regen_attachments):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"]) if isinstance(event["data"], dict) else event["data"],
            }

    return EventSourceResponse(event_generator())
