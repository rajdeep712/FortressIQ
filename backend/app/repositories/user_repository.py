from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_user_id(
        self,
        user_id: str,
    ) -> User | None:
        return self.db.execute(
            select(User).where(
                User.user_id == user_id
            )
        ).scalar_one_or_none()

    def get_by_email(
        self,
        email: str,
    ) -> User | None:
        return self.db.execute(
            select(User).where(
                User.email == email.lower()
            )
        ).scalar_one_or_none()

    def get_by_google_sub(
        self,
        google_sub: str,
    ) -> User | None:
        return self.db.execute(
            select(User).where(
                User.google_sub == google_sub
            )
        ).scalar_one_or_none()

    def get_by_password_reset_token(
        self,
        token_hash: str,
    ) -> User | None:
        return self.db.execute(
            select(User).where(
                User.password_reset_token == token_hash
            )
        ).scalar_one_or_none()

    def create(
        self,
        user: User,
    ) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(
        self,
        user: User,
    ) -> User:
        self.db.commit()
        self.db.refresh(user)
        return user

    def record_login(
        self,
        user: User,
    ) -> User:
        user.last_login_at = datetime.now(timezone.utc)
        return self.update(user)

    def set_password_reset_token(
        self,
        user: User,
        token_hash: str,
        expires_at: datetime,
    ) -> User:
        user.password_reset_token = token_hash
        user.password_reset_token_expire_date = expires_at
        return self.update(user)

    def clear_password_reset_token(
        self,
        user: User,
    ) -> User:
        user.password_reset_token = None
        user.password_reset_token_expire_date = None
        return self.update(user)

    def set_password(
        self,
        user: User,
        password_hash: str,
    ) -> User:
        user.password_hash = password_hash
        user.password_reset_token = None
        user.password_reset_token_expire_date = None
        return self.update(user)