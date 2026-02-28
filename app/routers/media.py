from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from app.models.message import MediaAttachment
from app.models.user import User
from app.storage.base import get_storage_backend

router = APIRouter()


@router.get("/{media_id}")
async def get_media(
    media_id: UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    result = await db.execute(
        select(MediaAttachment).where(MediaAttachment.id == media_id),
    )
    attachment = result.scalar_one_or_none()
    if attachment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found")

    storage = get_storage_backend()
    path = await storage.get_path(attachment.file_path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found on disk")

    return FileResponse(path, media_type=attachment.mime_type)


@router.get("/{media_id}/thumbnail")
async def get_thumbnail(
    media_id: UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    result = await db.execute(
        select(MediaAttachment).where(MediaAttachment.id == media_id),
    )
    attachment = result.scalar_one_or_none()
    if attachment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found")

    if not attachment.thumbnail_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No thumbnail available")

    storage = get_storage_backend()
    path = await storage.get_path(attachment.thumbnail_path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thumbnail file not found on disk")

    return FileResponse(path, media_type=attachment.mime_type)
