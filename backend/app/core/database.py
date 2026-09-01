from sqlalchemy import create_engine  ## creates the connection to your database.
from sqlalchemy.orm import sessionmaker  ## creates database sessions that you use to run queries.

from app.core.config import settings  ## import the configuration settings object.
from app.models.document import Base  ## contains your SQLAlchemy models/tables.


engine = create_engine(
    settings.database_url,
    connect_args={
        "check_same_thread": False  ## This is only required for sqlite, because SQLite normally restricts database access to the thread that created the connection.
    }
    if settings.database_url.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,  ## Prevents automatic session flushing before every query
    autocommit=False,  ## Prevents automatic session committing after every query
)


def init_db():
    Base.metadata.create_all(bind=engine)  ## Look at all my models and create their tables in the database if they don't already exist.

    ## create_all never alters existing tables, so newly added columns on
    ## already-created tables need a lightweight migration here.
    _add_column_if_missing(
        engine,
        table="users",
        column="avatar_url",
        definition="VARCHAR(512)",
    )
    _add_column_if_missing(
        engine,
        table="users",
        column="is_active",
        definition="BOOLEAN NOT NULL DEFAULT 1",
    )
    _add_column_if_missing(
        engine,
        table="users",
        column="last_login_at",
        definition="DATETIME",
    )
    _add_column_if_missing(
        engine,
        table="users",
        column="password_reset_token",
        definition="VARCHAR(128)",
    )
    _add_column_if_missing(
        engine,
        table="users",
        column="password_reset_token_expire_date",
        definition="DATETIME",
    )
    _add_column_if_missing(
        engine,
        table="refresh_tokens",
        column="used_at",
        definition="DATETIME",
    )
    _add_column_if_missing(
        engine,
        table="refresh_tokens",
        column="replaced_by_token_id",
        definition="INTEGER",
    )


def _add_column_if_missing(
    engine,
    table: str,
    column: str,
    definition: str,
):
    try:
        with engine.connect() as conn:
            existing = [
                row[1]
                for row in conn.exec_driver_sql(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            ]
    except Exception:
        # Table does not exist yet (create_all handles it above).
        return

    if column not in existing:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

## Provide sessions to FastAPI endpoints (get_db)
def get_db():
    db = SessionLocal() ## Create Database Session.

    try:
        yield db ## Provide session to endpoint.
    finally:
        db.close() ## Close session.