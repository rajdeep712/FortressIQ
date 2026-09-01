from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class DeviceMetadata(BaseModel):
    """Optional client-supplied info recorded with each session."""

    device_name: str | None = None
    platform: str | None = None
    browser: str | None = None


class RegisterRequest(DeviceMetadata):
    email: EmailStr
    password: str = Field(
        min_length=8,
        description="At least 8 characters.",
    )
    full_name: str = Field(
        min_length=1,
        description="User's display name.",
    )


class LoginRequest(DeviceMetadata):
    email: EmailStr
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(
        min_length=8,
        description="At least 8 characters.",
    )


class GoogleMergeRequest(DeviceMetadata):
    merge_token: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    email: str
    full_name: str | None
    avatar_url: str
    is_verified: bool
    provider: str
    created_at: datetime


class AuthResponse(BaseModel):
    """Returned on successful auth. Access + refresh tokens are delivered
    as httpOnly cookies, so the body carries only the user."""

    user: UserResponse


class MergeRequiredResponse(BaseModel):
    """Returned by the Google callback when a local account with the same
    email exists but has not yet been linked to the Google identity."""

    requires_merge: bool = True
    email: str
    name: str | None = None
    avatar_url: str | None = None
    merge_token: str


class SessionResponse(BaseModel):
    session_id: str
    device_name: str | None
    platform: str | None
    browser: str | None
    ip_address: str | None
    user_agent: str | None
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime


class VerifiedResponse(BaseModel):
    verified: bool
    user: UserResponse


class ResendResponse(BaseModel):
    sent: bool


class LogoutResponse(BaseModel):
    revoked: bool
