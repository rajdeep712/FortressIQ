from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.refresh_token import RefreshToken


class RefreshTokenRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_token_hash(
        self,
        token_hash: str,
    ) -> RefreshToken | None:
        return self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash
            )
        ).scalar_one_or_none()

    def create(
        self,
        token: RefreshToken,
    ) -> RefreshToken:
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

    def revoke(
        self,
        token: RefreshToken,
    ) -> RefreshToken:
        token.revoked_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(token)
        return token

    def mark_used(
        self,
        token: RefreshToken,
    ) -> RefreshToken:
        token.used_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(token)
        return token

    def mark_replaced(
        self,
        token: RefreshToken,
        replacement: RefreshToken,
    ) -> RefreshToken:
        token.replaced_by_token_id = replacement.id
        self.db.commit()
        self.db.refresh(token)
        return token

    def revoke_all_for_user(
        self,
        user_id: str,
    ) -> int:
        result = self.db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(timezone.utc))
        )
        self.db.commit()
        return result.rowcount

    def revoke_all_for_session(
        self,
        session_id: str,
    ) -> int:
        result = self.db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.session_id == session_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(timezone.utc))
        )
        self.db.commit()
        return result.rowcount

    def active_by_user(
        self,
        user_id: str,
    ) -> list[RefreshToken]:
        now = datetime.now(timezone.utc)
        return list(
            self.db.execute(
                select(RefreshToken)
                .where(
                    RefreshToken.user_id == user_id,
                    RefreshToken.revoked_at.is_(None),
                    RefreshToken.expires_at > now,
                )
                .order_by(RefreshToken.created_at.desc())
            ).scalars()
        )

    def active_by_session(
        self,
        session_id: str,
    ) -> list[RefreshToken]:
        now = datetime.now(timezone.utc)
        return list(
            self.db.execute(
                select(RefreshToken)
                .where(
                    RefreshToken.session_id == session_id,
                    RefreshToken.revoked_at.is_(None),
                    RefreshToken.expires_at > now,
                )
                .order_by(RefreshToken.created_at.desc())
            ).scalars()
        )