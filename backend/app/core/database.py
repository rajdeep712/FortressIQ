from sqlalchemy import create_engine  ## creates the connection to your database.
from sqlalchemy.orm import sessionmaker  ## creates database sessions that you use to run queries.

from app.core.config import settings  ## import the configuration settings object.
from app.models.document import Base  ## contains your SQLAlchemy models/tables.


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _engine_url(raw: str) -> str:
    """Normalize the configured URL for SQLAlchemy.

    SQLAlchemy only understands ``postgresql://`` / ``postgresql+driver://``;
    it cannot load a dialect for the ``postgres://`` scheme. Psycopg3 is the
    installed driver, so normalize ``postgres://`` -> ``postgresql+psycopg://``.
    """
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://"):]
    return raw


engine = create_engine(
    _engine_url(settings.database_url),
    connect_args={
        "check_same_thread": False  ## This is only required for sqlite, because SQLite normally restricts database access to the thread that created the connection.
    }
    if _is_sqlite(settings.database_url)
    else {},
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,  ## Prevents automatic session flushing before every query
    autocommit=False,  ## Prevents automatic session committing after every query
)


def init_db():
    import app.models  # noqa: F401  (register all models on the shared Base)
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
        definition="BOOLEAN NOT NULL DEFAULT TRUE",
    )
    _add_column_if_missing(
        engine,
        table="users",
        column="last_login_at",
        definition="TIMESTAMP",
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
        definition="TIMESTAMP",
    )
    _add_column_if_missing(
        engine,
        table="refresh_tokens",
        column="used_at",
        definition="TIMESTAMP",
    )
    _add_column_if_missing(
        engine,
        table="refresh_tokens",
        column="replaced_by_token_id",
        definition="INTEGER",
    )


def _column_names_postgres(engine, table: str):
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_name = %s",
            (table,),
        ).fetchall()
    return {row[0] for row in rows}


def _column_names_sqlite(engine, table: str):
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"PRAGMA table_info({table})"
        ).fetchall()
    return {row[1] for row in rows}


def _add_column_if_missing(
    engine,
    table: str,
    column: str,
    definition: str,
):
    is_sqlite = _is_sqlite(str(engine.url))

    try:
        existing = (
            _column_names_sqlite(engine, table)
            if is_sqlite
            else _column_names_postgres(engine, table)
        )
    except Exception:
        # Table does not exist yet (create_all handles it above).
        return

    if column not in existing:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )
        except Exception:
            # A concurrent `create_all` / another replica may have already
            # added the column; swallow idempotent collisions on Postgres.
            if not is_sqlite:
                return
            raise


## Provide sessions to FastAPI endpoints (get_db)
def get_db():
    db = SessionLocal() ## Create Database Session.

    try:
        yield db ## Provide session to endpoint.
    finally:
        db.close() ## Close session.
