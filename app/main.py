from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.media_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


from app.routers import auth, chat, media, threads  # noqa: E402

app.include_router(auth.router, prefix=f"{settings.api_v1_prefix}/auth", tags=["auth"])
app.include_router(threads.router, prefix=f"{settings.api_v1_prefix}/threads", tags=["threads"])
app.include_router(chat.router, prefix=f"{settings.api_v1_prefix}/chat", tags=["chat"])
app.include_router(media.router, prefix=f"{settings.api_v1_prefix}/media", tags=["media"])


@app.get("/health")
async def health():
    return {"status": "ok"}
