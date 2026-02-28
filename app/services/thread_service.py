from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message
from app.models.thread import Thread
from app.services.llm.router import get_provider

logger = logging.getLogger(__name__)

TITLE_SYSTEM_PROMPT = (
    "Generate a very short title (max 6 words) that summarizes the conversation below. "
    "Reply with ONLY the title text, no quotes, no punctuation at the end, no explanation."
)


async def generate_thread_title(db: AsyncSession, thread: Thread) -> str:
    """Use the thread's LLM to generate a concise title from the conversation."""
    stmt = (
        select(Message)
        .where(Message.thread_id == thread.id)
        .order_by(Message.created_at)
        .limit(10)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()

    if not messages:
        return thread.title

    conversation = [{"role": m.role, "content": m.content or ""} for m in messages]
    llm_messages = [
        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
        *conversation,
        {"role": "user", "content": "Generate a short title for this conversation."},
    ]

    try:
        provider = get_provider(thread.llm_name)
        chunks: list[str] = []
        async for chunk in provider.stream_completion(
            llm_messages, model=thread.llm_name, max_tokens=30,
        ):
            if isinstance(chunk, str):
                chunks.append(chunk)

        title = "".join(chunks).strip().strip('"').strip("'")
        if not title:
            return thread.title

        if len(title) > 100:
            title = title[:100]

        thread.title = title
        await db.commit()
        await db.refresh(thread)
        return title
    except Exception:
        logger.exception("Failed to generate thread title for %s", thread.id)
        return thread.title
