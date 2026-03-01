from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db
from app.models.message import MediaAttachment
from app.models.user import User
from app.storage.base import get_storage_backend

router = APIRouter()


async def _get_user_from_token(token: str, db: AsyncSession) -> User:
    """Validate a JWT access token and return the user."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@router.get("/{media_id}")
async def get_media(
    media_id: UUID,
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    await _get_user_from_token(token, db)

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
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    await _get_user_from_token(token, db)

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
