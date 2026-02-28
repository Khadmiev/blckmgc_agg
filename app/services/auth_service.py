from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import HTTPException, status
from jose import JWTError, jwt
import bcrypt
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.user import User
from app.schemas.auth import UserRegister

APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: UUID) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_access_token_expire_minutes,
    )
    payload = {"sub": str(user_id), "type": "access", "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: UUID) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.jwt_refresh_token_expire_days,
    )
    payload = {"sub": str(user_id), "type": "refresh", "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def register_user(db: AsyncSession, data: UserRegister) -> User:
    result = await db.execute(select(User).where(User.email == data.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email=data.email,
        username=data.username,
        hashed_password=hash_password(data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None or user.hashed_password is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return user


async def get_or_create_oauth_user(
    db: AsyncSession,
    email: str,
    username: str,
    provider: str,
    oauth_id: str,
    avatar_url: str | None,
) -> User:
    result = await db.execute(
        select(User).where(
            or_(
                (User.oauth_provider == provider) & (User.oauth_id == oauth_id),
                User.email == email,
            ),
        ),
    )
    user = result.scalar_one_or_none()

    if user is not None:
        if user.oauth_provider is None:
            user.oauth_provider = provider
            user.oauth_id = oauth_id
        if avatar_url and not user.avatar_url:
            user.avatar_url = avatar_url
        await db.commit()
        await db.refresh(user)
        return user

    user = User(
        email=email,
        username=username,
        oauth_provider=provider,
        oauth_id=oauth_id,
        avatar_url=avatar_url,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def verify_google_id_token(id_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": id_token},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token",
        )

    data = resp.json()
    if data.get("aud") != settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google token audience mismatch",
        )

    return {
        "email": data["email"],
        "name": data.get("name", data["email"].split("@")[0]),
        "picture": data.get("picture"),
        "sub": data["sub"],
    }


async def verify_apple_auth_code(code: str) -> dict:
    client = AsyncOAuth2Client(
        client_id=settings.apple_client_id,
        client_secret=settings.apple_private_key,
    )
    token = await client.fetch_token(
        APPLE_TOKEN_URL,
        grant_type="authorization_code",
        code=code,
    )

    id_token = token.get("id_token")
    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Apple auth failed: no id_token",
        )

    # Decode without verification — token was received directly from Apple over TLS.
    payload = jwt.get_unverified_claims(id_token)
    return {
        "email": payload.get("email", ""),
        "name": payload.get("email", "").split("@")[0],
        "picture": None,
        "sub": payload["sub"],
    }
