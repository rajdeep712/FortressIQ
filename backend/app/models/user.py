from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.models.document import Base


def generate_user_id() -> str:
    return f"user_{uuid4().hex}"


## Defines the 'users' table: accounts created by email+password or
## Google, uniquely keyed by user_id for tenancy across S3/Qdrant/DB.
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    password_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    # Google `sub` claim, once a Google identity has ever signed in.
    google_sub: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
        nullable=True,
    )

    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Profile picture: Google provides one (picture claim). Email/password
    # accounts stay NULL and get settings.default_avatar_url in responses.
    avatar_url: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    # How the account was first created: "email" or "google".
    provider: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="email",
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # Hard-disable switch (e.g. after abuse). Mirrors Node's isActive.
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Password-reset flow: only the SHA-256 digest of the reset token is
    # stored, with a short expiry window.
    password_reset_token: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    password_reset_token_expire_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )