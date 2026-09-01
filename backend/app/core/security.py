from datetime import datetime, timedelta, timezone
from hashlib import sha256
from secrets import token_urlsafe

import bcrypt
import jwt

from app.core.config import settings

ALGORITHM = "HS256"


def hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of a raw token.

    Used for refresh tokens and password-reset tokens so only the digest
    is ever persisted; a leaked DB never leaks usable secrets."""
    return sha256(token.encode("utf-8")).hexdigest()


def hash_password(
    password: str,
) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8"),
    )


def _encode_token(
    payload: dict,
    expires_delta: timedelta,
) -> str:
    now = datetime.now(timezone.utc)
    token_payload = {
        **payload,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(
        token_payload,
        settings.jwt_secret_key,
        algorithm=ALGORITHM,
    )


def create_access_token(
    user_id: str,
    email: str,
) -> str:
    return _encode_token(
        payload={
            "type": "access",
            "sub": user_id,
            "email": email,
        },
        expires_delta=timedelta(
            minutes=settings.access_token_expire_minutes
        ),
    )


def create_verification_token(
    user_id: str,
    email: str,
) -> str:
    return _encode_token(
        payload={
            "type": "verify_email",
            "sub": user_id,
            "email": email,
        },
        expires_delta=timedelta(
            hours=settings.verification_token_expire_hours
        ),
    )


# Refresh tokens are opaque, high-entropy, single-use random strings.
# Only their SHA-256 digest is persisted so a leaked DB never leaks
# usable tokens.
def generate_refresh_token() -> str:
    return token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hash_token(token)


def generate_reset_token() -> str:
    """Opaque, single-use token for the password-reset flow.

    The raw value goes in the emailed link; only its digest is stored."""
    return token_urlsafe(32)


def sign_google_merge_token(
    email: str,
    google_sub: str,
    avatar: str | None = None,
) -> str:
    """Short-lived token asserting that a Google identity should be
    linked to the given local (email/password) account after the user
    proves they own that account by entering its password."""
    return _encode_token(
        payload={
            "type": "google_merge",
            "email": email,
            "google_sub": google_sub,
            "avatar": avatar,
        },
        expires_delta=timedelta(minutes=10),
    )


def verify_google_merge_token(token: str) -> dict:
    """Validate and return a Google merge token payload. Raises
    jwt.PyJWTError for invalid/expired tokens or wrong type."""
    return decode_token(token, expected_type="google_merge")


def decode_token(
    token: str,
    expected_type: str | None = None,
) -> dict:
    """Decode and validate a JWT.

    Raises jwt.PyJWTError for any invalid/expired token, or when the
    token type does not match ``expected_type``."""
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[ALGORITHM],
    )

    if (
        expected_type is not None
        and payload.get("type") != expected_type
    ):
        raise jwt.InvalidTokenError(
            "Token type mismatch."
        )

    return payload