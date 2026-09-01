from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Integer,
    String,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.models.document import Base


## Defines the 'refresh_tokens' table: one active row per device/session.
## Tokens rotate (old row revoked, new row inherits the same session_id),
## so "logout from all devices" is a simple revoke-all-for-user.
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    # SHA-256 digest of the opaque refresh token (raw token never stored).
    token_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    # Identifies the device/session family across rotations.
    session_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    # Login device metadata captured at sign-in, shown in /sessions.
    device_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    platform: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    browser: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Set when the token is rotated, logged out, or logged out everywhere.
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Set when this token is exchanged during a rotation. A token that is
    # presented after it has already been used/revoked triggers family
    # revocation (reuse detection), matching the Node grace-period logic.
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # id of the replacement token created during rotation (link the chain).
    replaced_by_token_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )