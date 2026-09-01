import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_token
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.services.cookie_service import ACCESS_COOKIE

_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _verify_token(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        return decode_token(token, expected_type="access")
    except jwt.PyJWTError:
        return None


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        _bearer
    ),
    db: Session = Depends(get_db),
) -> User:
    # Prefer the Authorization bearer token, then fall back to the
    # httpOnly accessToken cookie (mirrors the Node setup where the cookie
    # carries the access token).
    payload = None
    if (
        credentials is not None
        and credentials.scheme.lower() == "bearer"
    ):
        payload = _verify_token(credentials.credentials)
    else:
        payload = _verify_token(request.cookies.get(ACCESS_COOKIE))

    if payload is None:
        raise _unauthorized("Not authenticated.")

    user = UserRepository(db).get_by_user_id(payload["sub"])

    if user is None:
        raise _unauthorized(
            "Invalid or expired token."
        )

    if not user.is_active:
        raise _unauthorized(
            "Account is disabled."
        )

    return user


def require_verified_user(
    user: User = Depends(get_current_user),
) -> User:
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email has not been verified.",
        )
    return user