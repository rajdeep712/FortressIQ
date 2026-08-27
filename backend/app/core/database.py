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

## Provide sessions to FastAPI endpoints (get_db)
def get_db():
    db = SessionLocal() ## Create Database Session.

    try:
        yield db ## Provide session to endpoint.
    finally:
        db.close() ## Close session.