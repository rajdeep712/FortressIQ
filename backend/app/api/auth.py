import functools
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.tracing import maybe_span
from app.models.user import User
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    AuthResponse,
    ForgotPasswordRequest,
    GoogleMergeRequest,
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    ResendResponse,
    ResendVerificationRequest,
    ResetPasswordRequest,
    SessionResponse,
    UserResponse,
    VerifiedResponse,
    VerifyEmailRequest,
)
from app.services.auth_service import (
    AccountMergeRequiredError,
    AuthService,
    DuplicateEmailError,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    InvalidGoogleTokenError,
    PasswordResetError,
    RefreshTokenError,
    VerificationTokenError,
)
from app.services.cookie_service import (
    REFRESH_COOKIE,
    clear_auth_cookies,
    set_auth_cookies,
)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Auth"],
)


def _capture_device(
    request: Request,
    body,
) -> dict:
    return {
        "device_name": getattr(body, "device_name", None),
        "platform": getattr(body, "platform", None),
        "browser": getattr(body, "browser", None),
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("User-Agent"),
    }


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url or settings.default_avatar_url,
        is_verified=user.is_verified,
        provider=user.provider,
        created_at=user.created_at,
    )


def _auth_response(user: User) -> AuthResponse:
    return AuthResponse(user=_user_response(user))


def _service(db: Session) -> AuthService:
    return AuthService(
        repository=UserRepository(db),
        refresh_repository=RefreshTokenRepository(db),
    )


def _trace_endpoint(name: str, **attributes):
    """Wrap an auth endpoint body in a Phoenix CHAIN span (no-op when off)."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attrs = dict(attributes)
            body = kwargs.get("body")
            if body is not None:
                email = getattr(body, "email", None)
                if email:
                    attrs["email"] = email
            request = kwargs.get("request")
            client = getattr(request, "client", None)
            if client is not None:
                attrs["request_ip"] = client.host
            with maybe_span(name, kind="CHAIN", **attrs):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
@_trace_endpoint(
    "auth.register",
    method="POST",
    path="/api/v1/auth/register",
)
def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    try:
        user, access_token, refresh_token = _service(db).register(
            email=body.email,
            password=body.password,
            full_name=body.full_name,
            device=_capture_device(request, body),
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    set_auth_cookies(response, access_token, refresh_token)
    return _auth_response(user)


@router.post(
    "/login",
    response_model=AuthResponse,
)
@_trace_endpoint(
    "auth.login",
    method="POST",
    path="/api/v1/auth/login",
)
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    try:
        user, access_token, refresh_token = _service(db).login(
            email=body.email,
            password=body.password,
            device=_capture_device(request, body),
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    except EmailNotVerifiedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    set_auth_cookies(response, access_token, refresh_token)
    return _auth_response(user)


@router.get(
    "/google",
    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
)
@_trace_endpoint(
    "auth.google.start",
    method="GET",
    path="/api/v1/auth/google",
)
def google_login_url(
    request: Request,
):
    """Begin the server-side Google OAuth2 flow by redirecting the user's
    browser to Google's consent screen."""
    try:
        from app.services.auth_service import google_auth_url

        url = google_auth_url()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    return Response(status_code=307, headers={"Location": url})


@router.get(
    "/google/callback",
)
@_trace_endpoint(
    "auth.google.callback",
    method="GET",
    path="/api/v1/auth/google/callback",
    provider="google",
)
def google_callback(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Handle Google's redirect back. Exchanges the code, then either
    logs the user in and redirects to the frontend, or (when a local
    account with the same email exists) redirects to the merge screen."""
    error = request.query_params.get("error")
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth error: {error}",
        )

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code.",
        )

    from app.services import auth_service as asvc

    try:
        claims = asvc.exchange_google_code(code)
        user, access_token, refresh_token = _service(db).google_oauth(
            claims,
            device=_capture_device(request, None),
        )
    except (InvalidGoogleTokenError, asvc.GoogleOAuthError) as exc:
        query = urlencode({"status": "error", "message": str(exc)})
        return RedirectResponse(
            f"{settings.frontend_url.rstrip('/')}/auth/google/callback?{query}",
            status_code=302,
        )
    except AccountMergeRequiredError as exc:
        query = urlencode(
            {
                "status": "merge",
                "token": exc.merge_token,
                "email": exc.email,
                "name": exc.name or "",
                "avatar": exc.avatar or "",
            }
        )
        return RedirectResponse(
            f"{settings.frontend_url.rstrip('/')}/auth/merge?{query}",
            status_code=302,
        )

    # Cookies are set on the shared origin (same-origin dev proxy), then the
    # browser is sent back to the frontend which restores the session.
    frontend = settings.frontend_url.rstrip("/")
    redirect = RedirectResponse(
        f"{frontend}/auth/google/callback?status=ok",
        status_code=302,
    )
    set_auth_cookies(redirect, access_token, refresh_token)
    return redirect


@router.post(
    "/google/merge",
    response_model=AuthResponse,
)
@_trace_endpoint(
    "auth.google.merge",
    method="POST",
    path="/api/v1/auth/google/merge",
    provider="google",
)
def google_merge(
    body: GoogleMergeRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    try:
        user, access_token, refresh_token = _service(db).merge_google_account(
            merge_token=body.merge_token,
            password=body.password,
            device=_capture_device(request, body),
        )
    except (InvalidGoogleTokenError, InvalidCredentialsError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    set_auth_cookies(response, access_token, refresh_token)
    return _auth_response(user)


@router.post(
    "/refresh",
    response_model=AuthResponse,
)
@_trace_endpoint(
    "auth.refresh",
    method="POST",
    path="/api/v1/auth/refresh",
)
def refresh_tokens(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token.",
        )
    try:
        user, access_token, new_refresh_token = _service(db).refresh(
            refresh_token,
            device=_capture_device(request, None),
        )
    except RefreshTokenError as exc:
        clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    set_auth_cookies(response, access_token, new_refresh_token)
    return _auth_response(user)


@router.post(
    "/logout",
    response_model=LogoutResponse,
)
@_trace_endpoint(
    "auth.logout",
    method="POST",
    path="/api/v1/auth/logout",
)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    revoked = bool(
        refresh_token and _service(db).logout(refresh_token)
    )
    clear_auth_cookies(response)
    return LogoutResponse(revoked=revoked)


@router.post(
    "/logout-all",
    response_model=LogoutResponse,
)
@_trace_endpoint(
    "auth.logout_all",
    method="POST",
    path="/api/v1/auth/logout-all",
)
def logout_all(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    revoked = _service(db).logout_all(current_user.user_id)
    clear_auth_cookies(response)
    return LogoutResponse(revoked=revoked > 0)


# Node-style alias for revoking every session.
@router.post(
    "/revoke-all-sessions",
    response_model=LogoutResponse,
)
@_trace_endpoint(
    "auth.revoke_all_sessions",
    method="POST",
    path="/api/v1/auth/revoke-all-sessions",
)
def revoke_all_sessions(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    revoked = _service(db).logout_all(current_user.user_id)
    clear_auth_cookies(response)
    return LogoutResponse(revoked=revoked > 0)


@router.get(
    "/sessions",
    response_model=list[SessionResponse],
)
@_trace_endpoint(
    "auth.sessions",
    method="GET",
    path="/api/v1/auth/sessions",
)
def sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = _service(db).active_sessions(current_user.user_id)
    return [
        SessionResponse(
            session_id=row.session_id,
            device_name=row.device_name,
            platform=row.platform,
            browser=row.browser,
            ip_address=row.ip_address,
            user_agent=row.user_agent,
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            expires_at=row.expires_at,
        )
        for row in rows
    ]


@router.post(
    "/forgot-password",
    response_model=ResendResponse,
)
@_trace_endpoint(
    "auth.forgot_password",
    method="POST",
    path="/api/v1/auth/forgot-password",
)
def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    sent = _service(db).forgot_password(
        body.email,
        device=_capture_device(request, body),
    )
    return ResendResponse(sent=sent)


@router.post(
    "/reset-password",
    response_model=ResendResponse,
)
@_trace_endpoint(
    "auth.reset_password",
    method="POST",
    path="/api/v1/auth/reset-password",
)
def reset_password(
    body: ResetPasswordRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    try:
        _service(db).reset_password(
            raw_token=body.token,
            new_password=body.password,
            device=_capture_device(request, body),
        )
    except PasswordResetError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    clear_auth_cookies(response)
    return ResendResponse(sent=True)


@router.post(
    "/verify-email",
    response_model=VerifiedResponse,
)
@_trace_endpoint(
    "auth.verify_email",
    method="POST",
    path="/api/v1/auth/verify-email",
)
def verify_email(
    body: VerifyEmailRequest,
    db: Session = Depends(get_db),
):
    try:
        user = _service(db).verify_email(body.token)
    except VerificationTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return VerifiedResponse(
        verified=user.is_verified,
        user=_user_response(user),
    )


@router.post(
    "/resend-verification",
    response_model=ResendResponse,
)
@_trace_endpoint(
    "auth.resend_verification",
    method="POST",
    path="/api/v1/auth/resend-verification",
)
def resend_verification(
    body: ResendVerificationRequest,
    db: Session = Depends(get_db),
):
    sent = _service(db).resend_verification(body.email)
    return ResendResponse(sent=sent)


@router.get(
    "/me",
    response_model=UserResponse,
)
@_trace_endpoint(
    "auth.me",
    method="GET",
    path="/api/v1/auth/me",
)
def me(
    current_user: User = Depends(get_current_user),
):
    return _user_response(current_user)
