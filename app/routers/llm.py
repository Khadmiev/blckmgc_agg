from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.models.user import User
from app.services.llm.router import list_available_models
from app.services.llm.status import provider_status_tracker

router = APIRouter()


@router.get("/providers")
async def get_providers(_user: User = Depends(get_current_user)):
    """Return status of all configured LLM providers with their models and availability."""
    return provider_status_tracker.get_all_statuses()


@router.get("/models")
async def get_models(_user: User = Depends(get_current_user)):
    """Return flat list of all models from configured providers."""
    return list_available_models()
