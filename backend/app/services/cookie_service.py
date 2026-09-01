from fastapi import Response

from app.core.config import settings

ACCESS_COOKIE = "accessToken"
REFRESH_COOKIE = "refreshToken"

# Match the Node reference: access token lives in the root path, refresh
# token is scoped to /api/auth (so it is only sent to auth endpoints).
REFRESH_COOKIE_PATH = "/api/v1/auth"

# Refresh tokens live ~REFRESH_TOKEN_EXPIRE_DAYS; access ~15 minutes.
def _access_max_age() -> int:
    return settings.access_token_expire_minutes * 60


def _refresh_max_age() -> int:
    return settings.refresh_token_expire_days * 24 * 60 * 60


def _secure() -> bool:
    return settings.environment.lower() == "production"


def set_access_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=token,
        max_age=_access_max_age(),
        httponly=True,
        secure=_secure(),
        samesite="lax",
        path="/",
    )


def set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        max_age=_refresh_max_age(),
        httponly=True,
        secure=_secure(),
        samesite="lax",
        path=REFRESH_COOKIE_PATH,
    )


def set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    set_access_cookie(response, access_token)
    set_refresh_cookie(response, refresh_token)


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(key=ACCESS_COOKIE, path="/")
    response.delete_cookie(key=REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)
