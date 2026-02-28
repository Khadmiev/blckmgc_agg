from __future__ import annotations

import base64
import logging
import mimetypes
import uuid
from typing import AsyncGenerator

from sqlalchemy import select

logger = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.message import MediaAttachment, Message
from app.models.thread import Thread
from app.services.llm.base import TokenUsage
from app.services.llm.router import get_provider
from app.services.llm.status import provider_status_tracker
from app.services.pricing_service import MediaCounts, compute_cost, get_current_price
from app.storage.base import StorageBackend

MAX_HISTORY_MESSAGES = 50

VISION_CAPABLE = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
    "claude-sonnet-4-20250514", "claude-haiku-4-20250414",
    "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-06-05",
}


async def load_thread_history(
    db: AsyncSession, thread: Thread, include_media: bool = True,
) -> list[dict]:
    """Load the most recent messages from a thread and format them for LLM consumption."""
    stmt = (
        select(Message)
        .where(Message.thread_id == thread.id)
        .order_by(Message.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
        .options(selectinload(Message.attachments))
    )
    result = await db.execute(stmt)
    messages = list(reversed(result.scalars().all()))

    formatted: list[dict] = []
    for msg in messages:
        if include_media and msg.attachments and thread.llm_name in VISION_CAPABLE:
            content_parts: list[dict] = []
            if msg.content:
                content_parts.append({"type": "text", "text": msg.content})
            for att in msg.attachments:
                if att.media_type == "image":
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"attachment://{att.id}"},
                    })
            formatted.append({"role": msg.role, "content": content_parts or msg.content})
        else:
            formatted.append({"role": msg.role, "content": msg.content or ""})
    return formatted


async def build_llm_messages(
    db: AsyncSession,
    thread: Thread,
    user_content: str | list[dict],
    storage: StorageBackend,
) -> list[dict]:
    """Build the full message list: history + current user message, resolving image attachments to base64."""
    history = await load_thread_history(db, thread)
    history.append({"role": "user", "content": user_content})
    if thread.llm_name in VISION_CAPABLE:
        history = await _resolve_attachments(db, history, storage)
    return history


async def _resolve_attachments(
    db: AsyncSession, messages: list[dict], storage: StorageBackend,
) -> list[dict]:
    """Replace attachment:// URLs with base64 data URLs for vision models."""
    resolved: list[dict] = []
    for msg in messages:
        if isinstance(msg["content"], list):
            parts: list[dict] = []
            for part in msg["content"]:
                if (
                    part.get("type") == "image_url"
                    and part.get("image_url", {}).get("url", "").startswith("attachment://")
                ):
                    att_id = part["image_url"]["url"].removeprefix("attachment://")
                    result = await db.execute(
                        select(MediaAttachment).where(MediaAttachment.id == uuid.UUID(att_id))
                    )
                    att = result.scalar_one_or_none()
                    if att:
                        path = await storage.get_path(att.file_path)
                        if path.is_file():
                            with open(path, "rb") as f:
                                data = base64.b64encode(f.read()).decode()
                            mime = att.mime_type or "image/jpeg"
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{data}"},
                            })
                            continue
                        else:
                            logger.warning("Attachment file missing from storage: %s", att.file_path)
                    else:
                        logger.warning("Attachment record not found: %s", att_id)
                    continue
                parts.append(part)
            resolved.append({"role": msg["role"], "content": parts})
        else:
            resolved.append(msg)
    return resolved


AUDIO_BYTES_PER_SECOND = 16_000  # ~128 kbps
VIDEO_BYTES_PER_SECOND = 1_000_000  # ~8 Mbps rough estimate


def _build_media_counts(attachments: list[MediaAttachment] | None) -> MediaCounts:
    if not attachments:
        return MediaCounts()
    images = 0
    audio_sec = 0.0
    video_sec = 0.0
    for att in attachments:
        if att.media_type == "image":
            images += 1
        elif att.media_type == "audio":
            audio_sec += att.file_size / AUDIO_BYTES_PER_SECOND
        elif att.media_type == "video":
            video_sec += att.file_size / VIDEO_BYTES_PER_SECOND
    return MediaCounts(
        image_count=images,
        audio_seconds=audio_sec,
        video_seconds=video_sec,
    )


async def stream_llm_response(
    db: AsyncSession,
    thread: Thread,
    user_content: str | list[dict],
    storage: StorageBackend,
    attachments: list[MediaAttachment] | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream LLM response chunks as dicts suitable for SSE serialization.

    Event types:
    - {"event": "chunk", "data": "<text>"}
    - {"event": "done", "data": {"content": "...", "model": "...", "message_id": "..."}}
    - {"event": "error", "data": "<error message>"}
    """
    try:
        provider = get_provider(thread.llm_name)
        messages = await build_llm_messages(db, thread, user_content, storage)

        full_response: list[str] = []
        usage: TokenUsage | None = None
        async for chunk in provider.stream_completion(messages, model=thread.llm_name):
            if isinstance(chunk, TokenUsage):
                usage = chunk
            else:
                full_response.append(chunk)
                yield {"event": "chunk", "data": chunk}

        content = "".join(full_response)
        provider_status_tracker.record_success(provider.provider_name())

        cost_usd = None
        if usage:
            pricing = await get_current_price(db, thread.llm_name)
            if pricing:
                media = _build_media_counts(attachments)
                cost_usd = compute_cost(usage, pricing, media)

        assistant_msg = Message(
            thread_id=thread.id,
            role="assistant",
            content=content,
            model=thread.llm_name,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            token_count=usage.total_tokens if usage else None,
            cost_usd=cost_usd,
        )
        db.add(assistant_msg)
        await db.commit()
        await db.refresh(assistant_msg)

        done_data: dict = {
            "content": content,
            "model": thread.llm_name,
            "message_id": str(assistant_msg.id),
        }
        if usage:
            done_data["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        if cost_usd is not None:
            done_data["cost_usd"] = float(cost_usd)

        yield {"event": "done", "data": done_data}
    except Exception as exc:
        try:
            p = get_provider(thread.llm_name)
            provider_status_tracker.record_failure(p.provider_name(), str(exc))
        except ValueError:
            pass
        yield {"event": "error", "data": str(exc)}


async def save_user_message(
    db: AsyncSession,
    thread: Thread,
    content: str,
    attachments: list[MediaAttachment] | None = None,
) -> Message:
    """Persist a user message and its attachments."""
    msg = Message(thread_id=thread.id, role="user", content=content)
    db.add(msg)
    await db.flush()

    if attachments:
        for att in attachments:
            att.message_id = msg.id
            db.add(att)

    await db.commit()
    await db.refresh(msg)
    return msg


async def process_uploaded_files(
    files_data: list[tuple[str, bytes, str]],
    storage: StorageBackend,
    thread_id: uuid.UUID,
) -> list[MediaAttachment]:
    """Save uploaded files and create MediaAttachment records (not yet linked to a message)."""
    attachments: list[MediaAttachment] = []
    subdir = str(thread_id)

    for original_name, data, content_type in files_data:
        ext = mimetypes.guess_extension(content_type) or ""
        filename = f"{uuid.uuid4().hex}{ext}"

        key = await storage.save(data, filename, subdir)

        media_type = "image" if content_type.startswith("image/") else \
                     "video" if content_type.startswith("video/") else \
                     "audio" if content_type.startswith("audio/") else "file"

        thumbnail_key = None
        if media_type == "image":
            try:
                thumbnail_key = await storage.save_thumbnail(data, filename, subdir)
            except Exception:
                pass

        att = MediaAttachment(
            media_type=media_type,
            file_path=key,
            mime_type=content_type,
            file_size=len(data),
            thumbnail_path=thumbnail_key,
        )
        attachments.append(att)

    return attachments
