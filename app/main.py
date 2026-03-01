from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings

logger = logging.getLogger(__name__)


def _db_host_from_url(url: str) -> str:
    """Extract host from DATABASE_URL for logging (no credentials)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.hostname or parsed.path.split("@")[-1].split("/")[0].split(":")[0] or "?"
    except Exception:
        return "?"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.media_dir).mkdir(parents=True, exist_ok=True)

    db_host = _db_host_from_url(settings.database_url)
    logger.info("Database host: %s", db_host)
    if "localhost" in settings.database_url or "127.0.0.1" in settings.database_url:
        logger.warning(
            "DATABASE_URL points to localhost. On Render/cloud, set DATABASE_URL to your Postgres Internal URL (Dashboard → Postgres → Connect)."
        )

    from app.services.llm.router import _get_model_map
    _get_model_map()

    from app.services.llm.status import provider_status_tracker
    logger.info("Running startup health checks for LLM providers...")
    await provider_status_tracker.check_all()

    logger.info("Fetching live model lists from LLM providers...")
    await provider_status_tracker.refresh_all_models()

    logger.info("Syncing LLM pricing from LiteLLM...")
    try:
        from app.database import async_session_factory
        from app.services.pricing_sync import sync_pricing
        async with async_session_factory() as db:
            result = await sync_pricing(db)
        logger.info(
            "Pricing sync complete: %d updated, %d unchanged",
            len(result.updated), result.unchanged,
        )
    except Exception:
        logger.exception("Pricing sync failed on startup")

    provider_status_tracker.start_background_checks()

    yield

    provider_status_tracker.stop_background_checks()


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


from app.routers import auth, chat, llm, media, pricing, threads  # noqa: E402

app.include_router(auth.router, prefix=f"{settings.api_v1_prefix}/auth", tags=["auth"])
app.include_router(threads.router, prefix=f"{settings.api_v1_prefix}/threads", tags=["threads"])
app.include_router(chat.router, prefix=f"{settings.api_v1_prefix}/chat", tags=["chat"])
app.include_router(media.router, prefix=f"{settings.api_v1_prefix}/media", tags=["media"])
app.include_router(llm.router, prefix=f"{settings.api_v1_prefix}/llm", tags=["llm"])
app.include_router(pricing.router, prefix=f"{settings.api_v1_prefix}/pricing", tags=["pricing"])


@app.get("/health")
async def health():
    """Basic health check. Use /health/db to verify database connectivity."""
    return {"status": "ok"}


@app.get("/health/db")
async def health_db():
    """Verify database connection. Returns 503 if DATABASE_URL is misconfigured or unreachable."""
    from sqlalchemy import text

    from app.database import async_session_factory

    try:
        async with async_session_factory() as db:
            await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "detail": "Database connection failed. Ensure DATABASE_URL is set in your environment (e.g. Render: add a PostgreSQL database and link it to this service).",
                "error": str(e),
            },
        )
    return {"status": "ok", "database": "connected"}


@app.get("/")
async def root():
    return FileResponse("web_client/index.html")


@app.get("/admin")
async def admin():
    return FileResponse("web_client/admin.html")


app.mount("/static", StaticFiles(directory="web_client"), name="static")
