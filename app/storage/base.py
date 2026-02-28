from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.config import settings


class StorageBackend(ABC):
    @abstractmethod
    async def save(self, file_data: bytes, filename: str, subdir: str = "") -> str: ...

    @abstractmethod
    async def get_path(self, key: str) -> Path: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    async def save_thumbnail(
        self,
        file_data: bytes,
        filename: str,
        subdir: str = "",
        size: tuple[int, int] = (256, 256),
    ) -> str:
        img = Image.open(BytesIO(file_data))
        img.thumbnail(size)
        buf = BytesIO()
        img.save(buf, format=img.format or "PNG")
        thumb_subdir = f"{subdir}/thumbs" if subdir else "thumbs"
        return await self.save(buf.getvalue(), filename, thumb_subdir)


def get_storage_backend() -> StorageBackend:
    if settings.storage_backend == "local":
        from app.storage.local import LocalStorage

        return LocalStorage()
    raise ValueError(f"Unknown storage backend: {settings.storage_backend!r}")
