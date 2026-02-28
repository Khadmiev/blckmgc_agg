from __future__ import annotations

from pathlib import Path

import aiofiles
import aiofiles.os

from app.config import settings
from app.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    def __init__(self) -> None:
        self._root = settings.media_path

    async def save(self, file_data: bytes, filename: str, subdir: str = "") -> str:
        dest_dir = self._root / subdir if subdir else self._root
        await aiofiles.os.makedirs(dest_dir, exist_ok=True)
        file_path = dest_dir / filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_data)
        return str(Path(subdir) / filename) if subdir else filename

    async def get_path(self, key: str) -> Path:
        return self._root / key

    async def delete(self, key: str) -> None:
        path = self._root / key
        try:
            await aiofiles.os.remove(path)
        except FileNotFoundError:
            pass
