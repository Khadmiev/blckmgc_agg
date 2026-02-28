from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from app.models.user import User
from app.schemas.auth import (
    OAuthAppleRequest,
    OAuthGoogleRequest,
    RefreshRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    UserUpdate,
)
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_or_create_oauth_user,
    register_user,
    verify_apple_auth_code,
    verify_google_id_token,
)

router = APIRouter()


def _token_pair(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/register", response_model=TokenResponse)
async def register(body: UserRegister, db: AsyncSession = Depends(get_db)):
    user = await register_user(db, body)
    return _token_pair(user)


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, body.email, body.password)
    return _token_pair(user)


@router.post("/oauth/google", response_model=TokenResponse)
async def oauth_google(body: OAuthGoogleRequest, db: AsyncSession = Depends(get_db)):
    info = await verify_google_id_token(body.id_token)
    user = await get_or_create_oauth_user(
        db,
        email=info["email"],
        username=info["name"],
        provider="google",
        oauth_id=info["sub"],
        avatar_url=info.get("picture"),
    )
    return _token_pair(user)


@router.post("/oauth/apple", response_model=TokenResponse)
async def oauth_apple(body: OAuthAppleRequest, db: AsyncSession = Depends(get_db)):
    info = await verify_apple_auth_code(body.code)
    user = await get_or_create_oauth_user(
        db,
        email=info["email"],
        username=info["name"],
        provider="apple",
        oauth_id=info["sub"],
        avatar_url=info.get("picture"),
    )
    return _token_pair(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    return TokenResponse(
        access_token=create_access_token(payload["sub"]),
        refresh_token=body.refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    await db.commit()
    await db.refresh(user)
    return user
