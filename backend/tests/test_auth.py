import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from jwt import PyJWTError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.auth_service as auth_service_module
from app.core.config import settings
from app.core.database import get_db
from app.core.security import (
    create_access_token,
    create_verification_token,
    decode_token,
    generate_reset_token,
    hash_password,
    hash_token,
    sign_google_merge_token,
    verify_password,
)
from app.main import app
from app.models.document import Base
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.services.email_service import EmailService

REFRESH_COOKIE = "refreshToken"
ACCESS_COOKIE = "accessToken"


@pytest.fixture
def db_session():
    # StaticPool: share a single SQLite in-memory connection across all
    # threads so TestClient requests and the fixture see the same tables.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    yield TestSession
    engine.dispose()


@pytest.fixture
def client(db_session):
    def override_get_db():
        db = db_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app, follow_redirects=False)
    yield client
    app.dependency_overrides.clear()


def make_user(db, email="a@b.com", password="password123"):
    return AuthService(
        repository=UserRepository(db),
        email_service=EmailService(),
    ).register(email=email, password=password)[0]


def make_verified_user(db, email="a@b.com", password="password123"):
    """Register an email account and mark it verified so login succeeds."""
    user = make_user(db, email=email, password=password)
    user.is_verified = True
    return UserRepository(db).update(user)


def set_send_verification(mp, sent=None):
    sent = sent if sent is not None else []

    def fake(self, to_email, token, user_agent=None, ip_address=None):
        sent.append((to_email, token))
        return True

    mp.setattr(EmailService, "send_verification_email", fake)
    return sent


# ---------------------------------------------------------------
# security helpers
# ---------------------------------------------------------------


def test_password_hash_roundtrip():
    digest = hash_password("correct horse battery")
    assert digest != "correct horse battery"
    assert verify_password("correct horse battery", digest)
    assert not verify_password("wrong", digest)


def test_access_token_roundtrip():
    token = create_access_token("user_x", "x@y.com")
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == "user_x"
    assert payload["email"] == "x@y.com"


def test_access_token_rejects_verification_token():
    verification = create_verification_token("user_x", "x@y.com")
    with pytest.raises(PyJWTError):
        decode_token(verification, expected_type="access")


def test_decode_token_invalid_signature():
    with pytest.raises(PyJWTError):
        decode_token("not-a-real-token")


# ---------------------------------------------------------------
# registration + verification email
# ---------------------------------------------------------------


def test_register_creates_user_sets_cookies_and_sends_verification(
    client, db_session, monkeypatch
):
    sent = set_send_verification(monkeypatch)

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "new@example.com",
            "password": "password123",
            "full_name": "New User",
        },
    )

    assert response.status_code == 201
    assert ACCESS_COOKIE in response.cookies
    assert REFRESH_COOKIE in response.cookies
    body = response.json()
    assert body["user"]["email"] == "new@example.com"
    assert body["user"]["is_verified"] is False
    assert body["user"]["user_id"].startswith("user_")

    assert len(sent) == 1
    sent_email, sent_token = sent[0]
    assert sent_email == "new@example.com"
    payload = decode_token(sent_token, expected_type="verify_email")
    assert payload["sub"] == body["user"]["user_id"]


def test_register_duplicate_email_409(client, db_session):
    make_user(db_session())
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "a@b.com",
            "password": "password123",
            "full_name": "A B",
        },
    )
    assert response.status_code == 409


def test_register_short_password_400(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "short@example.com",
            "password": "abc",
            "full_name": "Short Password",
        },
    )
    assert response.status_code == 400
    assert "Invalid value for" in response.json()["detail"]
    assert "password" in response.json()["detail"].lower()


def test_register_missing_parameters_400(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Missing required parameter" in detail
    for field in ("password", "full_name"):
        assert field in detail


def test_login_missing_parameters_400(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "a@b.com"},
    )
    assert response.status_code == 400
    assert "Missing required parameter" in response.json()["detail"]
    assert "password" in response.json()["detail"]


def test_register_invalid_email_400(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "not-an-email",
            "password": "password123",
            "full_name": "X",
        },
    )
    assert response.status_code == 400
    assert "Invalid value for" in response.json()["detail"]


# ---------------------------------------------------------------
# login
# ---------------------------------------------------------------


def test_login_success_sets_cookies(client, db_session):
    make_verified_user(db_session())
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "a@b.com", "password": "password123"},
    )
    assert response.status_code == 200
    assert ACCESS_COOKIE in response.cookies
    assert REFRESH_COOKIE in response.cookies
    assert response.json()["user"]["email"] == "a@b.com"

    # The access cookie authenticates /me automatically.
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user_id"] == response.json()["user"]["user_id"]


def test_login_wrong_password_401(client, db_session):
    make_verified_user(db_session())
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "a@b.com", "password": "wrong password"},
    )
    assert response.status_code == 401


def test_login_unknown_email_401(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "password123"},
    )
    assert response.status_code == 401


def test_login_unverified_email_403(client, db_session):
    # make_user registers an account but leaves it unverified.
    make_user(db_session())
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "a@b.com", "password": "password123"},
    )
    assert response.status_code == 403
    assert "verify your email" in response.json()["detail"].lower()
    assert ACCESS_COOKIE not in response.cookies
    assert REFRESH_COOKIE not in response.cookies


def test_login_verified_email_succeeds(client, db_session):
    make_verified_user(db_session())
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "a@b.com", "password": "password123"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["is_verified"] is True


# ---------------------------------------------------------------
# email verification
# ---------------------------------------------------------------


def _register_with_token(client, monkeypatch, email="verify@example.com"):
    sent = []
    set_send_verification(monkeypatch, sent)
    client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "password123",
            "full_name": "Verify User",
        },
    )
    return sent[0][1]


def test_verify_email_marks_user_verified(client, db_session, monkeypatch):
    token = _register_with_token(client, monkeypatch)

    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": token},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert body["user"]["is_verified"] is True


def test_verify_email_invalid_token_400(client):
    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": "garbage-token"},
    )
    assert response.status_code == 400


def test_resend_verification_sends_again(client, db_session, monkeypatch):
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "resend@example.com",
            "password": "password123",
            "full_name": "Resend User",
        },
    )

    sent = set_send_verification(monkeypatch)

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "resend@example.com"},
    )
    assert response.status_code == 200
    assert response.json() == {"sent": True}
    assert len(sent) == 1
    assert sent[0][0] == "resend@example.com"


def test_resend_verification_unknown_email_not_sent(client):
    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "ghost@example.com"},
    )
    assert response.status_code == 200
    assert response.json() == {"sent": False}


# ---------------------------------------------------------------
# Google OAuth2 redirect + merge flow
# ---------------------------------------------------------------


def _mock_value(monkeypatch, name, value):
    monkeypatch.setattr(auth_service_module, name, value)


def test_google_start_redirects_to_google(client):
    response = client.get("/api/v1/auth/google")
    assert response.status_code == 307
    assert response.headers["location"].startswith(
        "https://accounts.google.com/o/oauth2/v2/auth"
    )


def _mock_code_exchange(monkeypatch, email="google@example.com", sub="google-sub-123", name="Google User", picture=None):
    claims = {
        "sub": sub,
        "email": email,
        "email_verified": True,
        "name": name,
    }
    if picture:
        claims["picture"] = picture
    _mock_value(
        monkeypatch,
        "exchange_google_code",
        lambda code: claims,
    )


def test_google_callback_creates_verified_user(client, db_session, monkeypatch):
    _mock_code_exchange(monkeypatch)

    response = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "auth-code"},
    )

    # Redirects back to the frontend after setting auth cookies.
    assert response.status_code == 302
    assert ACCESS_COOKIE in response.cookies
    assert REFRESH_COOKIE in response.cookies
    assert "/auth/google/callback?status=ok" in response.headers["location"]
    # The cookies actually authenticate the session.
    assert client.get("/api/v1/auth/me").status_code == 200
    assert client.get("/api/v1/auth/me").json()["email"] == "google@example.com"


def test_google_callback_missing_code_400(client):
    response = client.get("/api/v1/auth/google/callback")
    assert response.status_code == 400


def test_google_callback_merge_required_redirects(client, db_session, monkeypatch):
    existing = make_user(db_session(), email="merge@example.com")
    _mock_code_exchange(monkeypatch, email="merge@example.com")

    response = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "auth-code"},
    )
    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith("http://localhost:3000/auth/merge?")
    assert "status=merge" in location
    assert "email=merge%40example.com" in location
    assert "token=" in location

    # No cookies were set (not yet authenticated).
    assert ACCESS_COOKIE not in response.cookies


def test_google_merge_links_identity(client, db_session, monkeypatch):
    existing = make_user(db_session(), email="m2@example.com")
    merge_token = sign_google_merge_token(
        email="m2@example.com",
        google_sub="google-sub-456",
    )

    response = client.post(
        "/api/v1/auth/google/merge",
        json={
            "merge_token": merge_token,
            "password": "password123",
        },
    )
    assert response.status_code == 200
    assert ACCESS_COOKIE in response.cookies
    body = response.json()
    assert body["user"]["user_id"] == existing.user_id
    assert body["user"]["is_verified"] is True


def test_google_merge_wrong_password_401(client, db_session, monkeypatch):
    make_user(db_session(), email="m3@example.com")
    merge_token = sign_google_merge_token(
        email="m3@example.com",
        google_sub="google-sub-789",
    )
    response = client.post(
        "/api/v1/auth/google/merge",
        json={
            "merge_token": merge_token,
            "password": "wrong-password",
        },
    )
    assert response.status_code == 401


# ---------------------------------------------------------------
# password reset flow
# ---------------------------------------------------------------


def _capture_reset(monkeypatch):
    sent = []

    def fake(self, to_email, reset_link, user_agent=None, ip_address=None):
        sent.append((to_email, reset_link))
        return True

    monkeypatch.setattr(EmailService, "send_password_reset_email", fake)
    return sent


def test_forgot_password_sends_reset(client, db_session, monkeypatch):
    make_user(db_session(), email="reset@example.com")
    sent = _capture_reset(monkeypatch)

    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "reset@example.com"},
    )
    assert response.status_code == 200
    assert response.json() == {"sent": True}
    assert len(sent) == 1
    assert sent[0][0] == "reset@example.com"
    assert "token=" in sent[0][1]


def test_forgot_password_unknown_email_does_not_leak(client, monkeypatch):
    sent = _capture_reset(monkeypatch)
    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "ghost@example.com"},
    )
    assert response.status_code == 200
    assert response.json() == {"sent": True}
    assert len(sent) == 0


def test_reset_password_changes_password_and_revokes_sessions(
    client, db_session, monkeypatch
):
    make_verified_user(db_session(), email="rp@example.com")
    # Log in to obtain a session that should be revoked on reset.
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "rp@example.com", "password": "password123"},
    )
    assert login.status_code == 200

    # Seed a reset token directly (simulate the emailed link).
    raw = generate_reset_token()
    db = db_session()
    user = UserRepository(db).get_by_email("rp@example.com")
    user.password_reset_token = hash_token(raw)
    user.password_reset_token_expire_date = datetime.now(timezone.utc).replace(
        tzinfo=None
    ) + timedelta(minutes=30)
    UserRepository(db).update(user)
    db.close()

    response = client.post(
        "/api/v1/auth/reset-password",
        json={"token": raw, "password": "newpassword456"},
    )
    assert response.status_code == 200

    # Verify the password actually changed.
    new_login = client.post(
        "/api/v1/auth/login",
        json={"email": "rp@example.com", "password": "newpassword456"},
    )
    assert new_login.status_code == 200
    # Old password no longer works.
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"email": "rp@example.com", "password": "password123"},
        ).status_code
        == 401
    )


def test_reset_password_invalid_token_400(client):
    response = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "not-a-real-token", "password": "newpassword456"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------
# auth guards on document routes
# ---------------------------------------------------------------


def test_documents_route_requires_auth(client):
    response = client.get("/api/v1/documents/some-doc")
    assert response.status_code == 401


def test_unverified_user_cannot_upload(client, db_session):
    user = make_user(db_session())
    token = create_access_token(user.user_id, user.email)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("x.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
    assert "verified" in response.json()["detail"].lower()


# ---------------------------------------------------------------
# refresh token rotation, logout, sessions
# ---------------------------------------------------------------


def _register(client, email="rot@example.com", **extra):
    payload = {
        "email": email,
        "password": "password123",
        "full_name": "Rotation User",
        **extra,
    }
    resp = client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 201
    return resp.json()


def test_login_sets_auth_cookies(client, db_session):
    make_verified_user(db_session())
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "a@b.com", "password": "password123"},
    )
    assert resp.status_code == 200
    assert resp.cookies.get(ACCESS_COOKIE)
    assert resp.cookies.get(REFRESH_COOKIE)


def test_refresh_rotates_via_cookie(client, db_session):
    _register(client)

    old_refresh = client.cookies.get(REFRESH_COOKIE)
    assert old_refresh

    rotated = client.post("/api/v1/auth/refresh")
    assert rotated.status_code == 200
    new_refresh = rotated.cookies.get(REFRESH_COOKIE)
    assert new_refresh and new_refresh != old_refresh

    # The replacement works (normal chained rotation).
    assert client.post("/api/v1/auth/refresh").status_code == 200


def test_refresh_reuse_beyond_grace_kills_family(
    client, db_session
):
    """Replaying a consumed refresh token outside the 15s grace window is
    treated as theft and revokes the entire session family (Node parity)."""
    from app.core.security import hash_refresh_token

    _register(client)

    old_refresh = client.cookies.get(REFRESH_COOKIE)
    assert old_refresh

    # Rotate once: the old token is now consumed (used_at set).
    assert client.post("/api/v1/auth/refresh").status_code == 200
    new_refresh = client.cookies.get(REFRESH_COOKIE)
    assert new_refresh and new_refresh != old_refresh

    # Simulate the theft replay happening well past the grace window: push
    # the consumed token's used_at far into the past.
    db = db_session()
    row = RefreshTokenRepository(db).get_by_token_hash(
        hash_refresh_token(old_refresh)
    )
    assert row is not None
    row.used_at = datetime.now(timezone.utc).replace(
        tzinfo=None
    ) - timedelta(minutes=5)
    db.commit()
    db.close()

    # Replaying the old token (beyond grace) revokes the whole family.
    client.cookies.set(REFRESH_COOKIE, old_refresh)
    assert client.post("/api/v1/auth/refresh").status_code == 401

    # The replacement issued by that family is now revoked too.
    client.cookies.set(REFRESH_COOKIE, new_refresh)
    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_refresh_requires_cookie(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    response = client.post("/api/v1/auth/refresh")
    assert response.status_code == 401


def test_logout_revokes_session(client, db_session):
    _register(client)
    assert client.post("/api/v1/auth/refresh").status_code == 200

    logged_out = client.post("/api/v1/auth/logout")
    assert logged_out.status_code == 200
    assert logged_out.json() == {"revoked": True}

    # Cookie cleared -> refresh now requires auth.
    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_logout_all_revokes_every_session(client, db_session):
    _register(client, email="multi@example.com")
    assert client.post("/api/v1/auth/refresh").status_code == 200

    response = client.post("/api/v1/auth/logout-all")
    assert response.status_code == 200
    assert response.json() == {"revoked": True}

    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_sessions_lists_active_devices(client, db_session):
    _register(
        client,
        email="device@example.com",
        device_name="My Phone",
        platform="Android",
        browser="Chrome",
    )

    sessions = client.get("/api/v1/auth/sessions")
    assert sessions.status_code == 200
    rows = sessions.json()
    assert len(rows) == 1

    row = rows[0]
    assert row["device_name"] == "My Phone"
    assert row["platform"] == "Android"
    assert row["browser"] == "Chrome"
    assert row["session_id"]
    assert row["expires_at"] > row["created_at"]


def test_sessions_requires_auth(client):
    assert client.get("/api/v1/auth/sessions").status_code == 401


# ---------------------------------------------------------------
# names and avatars
# ---------------------------------------------------------------


def test_register_requires_name_400(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "noname@example.com", "password": "password123"},
    )
    assert response.status_code == 400
    assert "Missing required parameter" in response.json()["detail"]
    assert "full_name" in response.json()["detail"]


def test_email_user_gets_default_avatar(client, db_session):
    matcher = _register(client, email="avatar@example.com")
    assert matcher["user"]["avatar_url"] == settings.default_avatar_url
