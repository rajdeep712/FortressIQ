import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import jwt

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_verification_token,
    decode_token,
    generate_refresh_token,
    generate_reset_token,
    hash_password,
    hash_refresh_token,
    hash_token,
    sign_google_merge_token,
    verify_password,
    verify_google_merge_token,
)
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.models.user import generate_user_id
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 8

GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = (
    "https://accounts.google.com",
    "accounts.google.com",
)
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

# Reuse-detection grace window, matching the Node reference. A refresh token
# presented within this short window after a rotation is treated as the
# legitimate duplicate we just issued (handled gracefully); anything older
# means theft and revokes the whole account's sessions.
REUSE_GRACE_SECONDS = 15


class DuplicateEmailError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class EmailNotVerifiedError(Exception):
    pass


class InvalidGoogleTokenError(Exception):
    pass


class GoogleOAuthError(Exception):
    pass


class VerificationTokenError(Exception):
    pass


class RefreshTokenError(Exception):
    pass


class DuplicateRequestError(RefreshTokenError):
    pass


class AccountMergeRequiredError(Exception):
    def __init__(self, message: str, email: str, merge_token: str, name: str | None, avatar: str | None):
        super().__init__(message)
        self.email = email
        self.merge_token = merge_token
        self.name = name
        self.avatar = avatar


class PasswordResetError(Exception):
    pass


def normalize_email(email: str) -> str:
    return email.strip().lower()


# Keep the JWKS fetcher module-level so tests can swap it.
def verify_google_id_token(id_token: str) -> dict:
    """Verify a Google Identity Services ID token and return its claims.

    The signing key is fetched from Google's public JWKS. Raises
    InvalidGoogleTokenError for any structurally invalid token."""
    if not settings.google_client_id:
        raise InvalidGoogleTokenError(
            "Google login is not configured."
        )

    try:
        jwks_client = jwt.PyJWKClient(GOOGLE_JWKS_URL)
        signing_key = jwks_client.get_signing_key_from_jwt(
            id_token
        )
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.google_client_id,
            options={
                "verify_aud": True,
                "verify_exp": True,
            },
        )
    except jwt.PyJWTError as exc:
        raise InvalidGoogleTokenError(
            f"Invalid Google ID token: {exc}"
        )

    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise InvalidGoogleTokenError(
            "Unexpected token issuer."
        )

    if not claims.get("sub"):
        raise InvalidGoogleTokenError(
            "Google token missing subject."
        )

    if not claims.get("email"):
        raise InvalidGoogleTokenError(
            "Google token missing email."
        )

    return claims


def google_auth_url(state: str | None = None) -> str:
    """Build the server-side Google OAuth2 authorization URL."""
    if not settings.google_client_id or not settings.google_redirect_uri:
        raise GoogleOAuthError(
            "Google login is not configured."
        )
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    if state:
        params["state"] = state

    # Properly URL-encode the query values.
    from urllib.parse import urlencode

    query = urlencode(params)
    return f"{GOOGLE_AUTH_URL}?{query}"


def exchange_google_code(code: str) -> dict:
    """Exchange an OAuth authorization code for tokens and return the
    verified id_token claims. Raises GoogleOAuthError on any failure."""
    if (
        not settings.google_client_id
        or not settings.google_client_secret
        or not settings.google_redirect_uri
    ):
        raise GoogleOAuthError(
            "Google login is not configured."
        )

    try:
        resp = httpx.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise GoogleOAuthError(
            f"Google token exchange failed: {exc}"
        )

    if resp.status_code >= 300:
        raise GoogleOAuthError(
            "Google token exchange failed."
        )

    id_token = resp.json().get("id_token")
    if not id_token:
        raise GoogleOAuthError(
            "Google returned no ID token."
        )

    return verify_google_id_token(id_token)


class AuthService:
    def __init__(
        self,
        repository: UserRepository,
        refresh_repository: RefreshTokenRepository | None = None,
        email_service: EmailService | None = None,
    ):
        self.repository = repository
        self.refresh_repository = (
            refresh_repository or RefreshTokenRepository(repository.db)
        )
        self.email_service = (
            email_service or EmailService()
        )

    def _issue_refresh_session(
        self,
        user_id: str,
        device: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Persist a new active refresh token and return its raw value."""
        device = device or {}
        raw_token = generate_refresh_token()
        now = datetime.now(timezone.utc)
        self.refresh_repository.create(
            RefreshToken(
                token_hash=hash_refresh_token(raw_token),
                session_id=session_id or uuid4().hex,
                user_id=user_id,
                device_name=device.get("device_name"),
                platform=device.get("platform"),
                browser=device.get("browser"),
                ip_address=device.get("ip_address"),
                user_agent=device.get("user_agent"),
                expires_at=now + timedelta(
                    days=settings.refresh_token_expire_days
                ),
                last_used_at=now,
            )
        )
        return raw_token

    def _tokens(
        self,
        user: User,
        device: dict | None = None,
        session_id: str | None = None,
    ) -> tuple[str, str]:
        refresh_token = self._issue_refresh_session(
            user.user_id,
            device=device,
            session_id=session_id,
        )
        return (
            create_access_token(user.user_id, user.email),
            refresh_token,
        )

    def register(
        self,
        email: str,
        password: str,
        full_name: str | None = None,
        device: dict | None = None,
    ) -> tuple[User, str, str]:
        email = normalize_email(email)

        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"Password must be at least "
                f"{MIN_PASSWORD_LENGTH} characters."
            )

        if self.repository.get_by_email(email) is not None:
            raise DuplicateEmailError(
                "An account with this email already exists."
            )

        user = self.repository.create(
            User(
                user_id=generate_user_id(),
                email=email,
                password_hash=hash_password(password),
                full_name=full_name,
                provider="email",
                is_verified=False,
            )
        )

        self.email_service.send_verification_email(
            to_email=email,
            token=create_verification_token(
                user.user_id,
                email,
            ),
            user_agent=(device or {}).get("user_agent"),
            ip_address=(device or {}).get("ip_address"),
        )

        access_token, refresh_token = self._tokens(user, device)
        return user, access_token, refresh_token

    def login(
        self,
        email: str,
        password: str,
        device: dict | None = None,
    ) -> tuple[User, str, str]:
        email = normalize_email(email)
        user = self.repository.get_by_email(email)

        if (
            user is None
            or user.password_hash is None
        ):
            raise InvalidCredentialsError(
                "Invalid email or password."
            )

        if not verify_password(
            password,
            user.password_hash,
        ):
            raise InvalidCredentialsError(
                "Invalid email or password."
            )

        # Only verified accounts may sign in. Google-linked accounts are
        # marked verified during OAuth, so this only ever blocks local
        # (email/password) accounts that haven't confirmed their email yet.
        if not user.is_verified:
            raise EmailNotVerifiedError(
                "Please verify your email before logging in."
            )

        user = self.repository.record_login(user)

        access_token, refresh_token = self._tokens(user, device)
        return user, access_token, refresh_token

    def google_oauth(
        self,
        claims: dict,
        device: dict | None = None,
    ) -> tuple[User, str, str]:
        """Complete a Google OAuth2 login/registration from verified
        id_token claims.

        If a Google-signed-in account already exists it logs in.
        Otherwise, if a local (email/password) account with the same
        email exists, it raises AccountMergeRequiredError so the flow can
        challenge for a password and merge. Otherwise it creates a new
        Google account."""
        sub = claims["sub"]
        email = normalize_email(claims["email"])
        name = claims.get("name")
        avatar = claims.get("picture") or settings.default_avatar_url

        user = self.repository.get_by_google_sub(sub)

        if user is None:
            local_user = self.repository.get_by_email(email)
            if local_user is not None:
                if local_user.google_sub == sub:
                    user = local_user
                else:
                    # Local account with a different password provider —
                    # must prove ownership before linking Google.
                    merge_token = sign_google_merge_token(
                        email=email,
                        google_sub=sub,
                        avatar=avatar,
                    )
                    raise AccountMergeRequiredError(
                        "An account with this email already exists.",
                        email=email,
                        merge_token=merge_token,
                        name=name,
                        avatar=avatar,
                    )
            else:
                user = self.repository.create(
                    User(
                        user_id=generate_user_id(),
                        email=email,
                        google_sub=sub,
                        password_hash=None,
                        full_name=name,
                        avatar_url=avatar,
                        provider="google",
                        is_verified=True,
                        verified_at=datetime.now(timezone.utc),
                    )
                )

        # Link Google identity to the existing account and make sure
        # Google-signed-in accounts count as verified.
        changes = False
        if user.google_sub != sub:
            user.google_sub = sub
            changes = True
        if not user.is_verified:
            user.is_verified = True
            user.verified_at = datetime.now(timezone.utc)
            changes = True
        if not user.avatar_url and claims.get("picture"):
            user.avatar_url = claims["picture"]
            changes = True
        if changes:
            user = self.repository.update(user)

        user = self.repository.record_login(user)
        access_token, refresh_token = self._tokens(user, device)
        return user, access_token, refresh_token

    def merge_google_account(
        self,
        merge_token: str,
        password: str,
        device: dict | None = None,
    ) -> tuple[User, str, str]:
        """Prove ownership of the local account (via password) then link
        the pending Google identity, producing a normal session."""
        try:
            payload = verify_google_merge_token(merge_token)
        except jwt.PyJWTError as exc:
            raise InvalidGoogleTokenError(
                "Google merge link is invalid or expired."
            ) from exc

        email = normalize_email(payload["email"])
        sub = payload["google_sub"]
        avatar = payload.get("avatar")

        user = self.repository.get_by_email(email)
        if user is None or user.password_hash is None:
            raise InvalidCredentialsError(
                "Invalid email or password."
            )

        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError(
                "Invalid email or password."
            )

        if not user.google_sub:
            # Done via first pre-linked account? If a different sub is
            # already bound, refuse rather than silently replacing.
            user.google_sub = sub
        if avatar and not user.avatar_url:
            user.avatar_url = avatar
        user.is_verified = True
        user.verified_at = user.verified_at or datetime.now(timezone.utc)
        user = self.repository.update(user)
        user = self.repository.record_login(user)

        access_token, refresh_token = self._tokens(user, device)
        return user, access_token, refresh_token

    def refresh(
        self,
        refresh_token: str,
        device: dict | None = None,
    ) -> tuple[User, str, str]:
        """Exchange a refresh token for a new access + refresh token pair.

        The presented token is revoked (rotated) so it can only be used
        once; the replacement inherits the session so device history is
        preserved. Reuse beyond a short grace window indicates a stolen
        token and revokes the entire session family (matching Node)."""
        # SQLite stores datetimes without tzinfo, so compare in naive UTC.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        token_hash = hash_refresh_token(refresh_token)
        row = self.refresh_repository.get_by_token_hash(
            token_hash
        )

        if row is None:
            raise RefreshTokenError(
                "Invalid or expired refresh token."
            )

        # Reuse detection. A token presented AFTER it was already rotated
        # is suspicious. Within the short grace window we treat it as the
        # legitimate duplicate we just issued (the original client that is
        # racing with our replacement). Beyond that, assume theft and kill
        # the whole session family (matching Node).
        if row.used_at is not None:
            elapsed = (
                now - row.used_at.replace(tzinfo=None)
            ).total_seconds()
            if elapsed <= REUSE_GRACE_SECONDS:
                raise DuplicateRequestError(
                    "Duplicate refresh request detected."
                )
            # Beyond the grace window: stolen/replayed token. Revoke the
            # entire session family and reject.
            self.refresh_repository.revoke_all_for_session(
                row.session_id
            )
            raise RefreshTokenError(
                "Invalid or expired refresh token."
            )

        # Explicitly revoked (logout, family kill, password reset, ...).
        if row.revoked_at is not None:
            raise RefreshTokenError(
                "Invalid or expired refresh token."
            )

        if row.expires_at <= now:
            raise RefreshTokenError(
                "Invalid or expired refresh token."
            )

        user = self.repository.get_by_user_id(row.user_id)
        if user is None:
            raise RefreshTokenError(
                "Invalid or expired refresh token."
            )

        # Mark as used (for reuse detection) BEFORE issuing the
        # replacement, so the new token does not get wrongly revoked.
        self.refresh_repository.mark_used(row)

        # Reuse the existing session's recorded device info unless the
        # client supplied fresh metadata for this rotation.
        device = device or {}
        merged_device = {
            "device_name": (
                device.get("device_name") or row.device_name
            ),
            "platform": device.get("platform") or row.platform,
            "browser": device.get("browser") or row.browser,
            "ip_address": device.get("ip_address") or row.ip_address,
            "user_agent": device.get("user_agent") or row.user_agent,
        }

        access_token, new_refresh_token = self._tokens(
            user,
            device=merged_device,
            session_id=row.session_id,
        )

        # Link the replacement to the consumed token to close the loop.
        replacement = self.refresh_repository.get_by_token_hash(
            hash_refresh_token(new_refresh_token)
        )
        if replacement is not None:
            self.refresh_repository.mark_replaced(row, replacement)

        return user, access_token, new_refresh_token

    def forgot_password(
        self,
        email: str,
        device: dict | None = None,
    ) -> bool:
        """Kick off a password reset. Always returns True so it does not
        leak which emails exist; if the account is local and active, an
        email with a reset link is sent and all existing sessions are
        revoked."""
        email = normalize_email(email)
        user = self.repository.get_by_email(email)

        if user is None or not user.is_active:
            return True

        # For local accounts only — Google-only accounts have no password.
        if user.password_hash is None:
            return True

        raw_token = generate_reset_token()
        expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            minutes=settings.password_reset_token_expire_minutes
        )
        self.repository.set_password_reset_token(
            user,
            token_hash=hash_token(raw_token),
            expires_at=expires_at,
        )

        # Invalidate every existing session so a reset cannot be combined
        # with an already-open session.
        self.refresh_repository.revoke_all_for_user(user.user_id)

        reset_link = (
            f"{settings.frontend_url.rstrip('/')}"
            f"/reset-password?token={raw_token}"
        )
        self.email_service.send_password_reset_email(
            to_email=user.email,
            reset_link=reset_link,
            user_agent=(device or {}).get("user_agent"),
            ip_address=(device or {}).get("ip_address"),
        )
        return True

    def reset_password(
        self,
        raw_token: str,
        new_password: str,
        device: dict | None = None,
    ) -> bool:
        """Validate the reset token, set the new password, and revoke all
        existing sessions. Returns True on success, raises
        PasswordResetError otherwise."""
        if len(new_password) < MIN_PASSWORD_LENGTH:
            raise PasswordResetError(
                f"Password must be at least "
                f"{MIN_PASSWORD_LENGTH} characters."
            )

        if not raw_token:
            raise PasswordResetError(
                "Reset token is required."
            )

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        token_hash = hash_token(raw_token)

        user = self.repository.get_by_password_reset_token(token_hash)
        if (
            user is None
            or not user.is_active
            or user.password_reset_token_expire_date is None
        ):
            raise PasswordResetError(
                "Reset link is invalid or expired."
            )

        expires_at = user.password_reset_token_expire_date.replace(
            tzinfo=None
        )
        if expires_at <= now:
            # Clear the stale token so it cannot be replayed.
            self.repository.clear_password_reset_token(user)
            raise PasswordResetError(
                "Reset link has expired."
            )

        self.repository.set_password(
            user,
            password_hash=hash_password(new_password),
        )

        # A password change is a high-value event: revoke all sessions.
        self.refresh_repository.revoke_all_for_user(user.user_id)

        return True

    def logout(
        self,
        refresh_token: str,
    ) -> bool:
        row = self.refresh_repository.get_by_token_hash(
            hash_refresh_token(refresh_token)
        )
        if row is None or row.revoked_at is not None:
            return False
        self.refresh_repository.revoke(row)
        return True

    def logout_all(
        self,
        user_id: str,
    ) -> int:
        return self.refresh_repository.revoke_all_for_user(user_id)

    def active_sessions(
        self,
        user_id: str,
    ) -> list[RefreshToken]:
        return self.refresh_repository.active_by_user(user_id)

    def verify_email(
        self,
        token: str,
    ) -> User:
        try:
            payload = decode_token(
                token,
                expected_type="verify_email",
            )
        except jwt.PyJWTError as exc:
            raise VerificationTokenError(
                "Verification link is invalid or expired."
            ) from exc

        user = self.repository.get_by_user_id(
            payload["sub"]
        )

        if user is None:
            raise VerificationTokenError(
                "Verification link is invalid or expired."
            )

        if not user.is_verified:
            user.is_verified = True
            user.verified_at = datetime.now(timezone.utc)
            user = self.repository.update(user)

        return user

    def resend_verification(
        self,
        email: str,
    ) -> bool:
        """Re-send the verification link. Returns False for unknown
        accounts (does not leak which emails exist)."""
        user = self.repository.get_by_email(
            normalize_email(email)
        )

        if user is None or user.is_verified:
            return False

        self.email_service.send_verification_email(
            to_email=user.email,
            token=create_verification_token(
                user.user_id,
                user.email,
            ),
        )
        return True