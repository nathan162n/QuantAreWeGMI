"""Database connection factory with automatic Supabase to SQLite fallback.

Provides a resilient database connection that tries Supabase PostgreSQL
first (via PgBouncer connection pooler) and automatically falls back to
local SQLite if Supabase is unreachable.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base
from src.utils.logger import get_logger

logger = get_logger("database.engine")

ACTIVE_DB_BACKEND: str = "uninitialized"

_engine: Engine = None  # type: ignore[assignment]
_SessionFactory: sessionmaker = None  # type: ignore[assignment]


def create_engine_with_fallback() -> Engine:
    """Attempt to connect to Supabase PostgreSQL, fall back to SQLite on failure.

    Priority:
      1. SUPABASE_POOLER_URL (PgBouncer - preferred for long-running 24/7 processes)
      2. DATABASE_URL (direct Supabase Postgres URL - fallback if pooler not configured)
      3. sqlite:///data/algotrader.db (local fallback - always works, 50GB available)

    Logs which connection was successfully established at INFO level.
    Logs WARNING if falling back to SQLite.
    Never raises - always returns a working engine.

    Returns:
        Engine: SQLAlchemy engine connected to the best available database.

    Example:
        >>> engine = create_engine_with_fallback()
        >>> # Engine is now connected to Supabase or SQLite
    """
    global ACTIVE_DB_BACKEND, _engine, _SessionFactory

    pool_size = int(os.environ.get("DB_POOL_SIZE", "5"))
    max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "10"))
    pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "1800"))
    connect_timeout = int(os.environ.get("DB_CONNECT_TIMEOUT", "10"))

    postgres_urls = []
    pooler_url = os.environ.get("SUPABASE_POOLER_URL", "")
    if pooler_url and pooler_url != "postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres":
        postgres_urls.append(("supabase_pooler", pooler_url))

    database_url = os.environ.get("DATABASE_URL", "")
    if database_url and database_url != "postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres":
        postgres_urls.append(("supabase_direct", database_url))

    for label, url in postgres_urls:
        try:
            logger.info("Attempting PostgreSQL connection via %s...", label)
            engine = create_engine(
                url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": connect_timeout,
                    "application_name": "algotrader",
                },
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()

            ACTIVE_DB_BACKEND = "supabase"
            _engine = engine
            _SessionFactory = sessionmaker(
                bind=engine, autocommit=False, autoflush=False
            )
            logger.info(
                "Successfully connected to Supabase PostgreSQL via %s", label
            )
            return engine
        except Exception as e:
            logger.warning(
                "Failed to connect via %s: %s. Trying next option...",
                label,
                str(e),
            )

    logger.warning(
        "All PostgreSQL connections failed. Falling back to local SQLite."
    )
    return _create_sqlite_engine()


def _create_sqlite_engine() -> Engine:
    """Create a SQLite engine as the fallback database.

    Uses StaticPool for thread safety with SQLite's single-writer limitation.
    Creates the data directory if it doesn't exist.

    Returns:
        Engine: SQLAlchemy engine connected to local SQLite database.

    Example:
        >>> engine = _create_sqlite_engine()
        >>> # Engine connected to data/algotrader.db
    """
    global ACTIVE_DB_BACKEND, _engine, _SessionFactory

    sqlite_url = os.environ.get("SQLITE_FALLBACK_URL", "sqlite:///data/algotrader.db")
    os.makedirs("data", exist_ok=True)

    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    ACTIVE_DB_BACKEND = "sqlite"
    _engine = engine
    _SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    logger.info("Connected to local SQLite database: %s", sqlite_url)
    return engine


def _migrate_sqlite_columns(engine: Engine) -> None:
    """Add any missing columns to existing SQLite tables.

    SQLAlchemy's create_all() does not ALTER existing tables. This
    function inspects each table and issues ALTER TABLE ADD COLUMN
    for any columns present in the ORM model but absent in the DB.

    Only runs for SQLite — PostgreSQL (Supabase) uses proper migrations.

    Args:
        engine: The active database engine.
    """
    if "sqlite" not in str(engine.url):
        return

    from sqlalchemy import inspect, Integer, Float, Boolean, String, Text, Numeric, DateTime, JSON

    inspector = inspect(engine)
    type_defaults = {
        "INTEGER": "0",
        "FLOAT": "0.0",
        "REAL": "0.0",
        "BOOLEAN": "0",
        "NUMERIC": "0",
        "VARCHAR": "''",
        "TEXT": "''",
    }

    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing_cols = {col["name"] for col in inspector.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing_cols:
                    continue
                col_type = str(col.type).upper().split("(")[0]
                default = type_defaults.get(col_type, "NULL")
                if col.nullable and default == "NULL":
                    ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type} DEFAULT NULL"
                else:
                    ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type} DEFAULT {default}"
                try:
                    conn.execute(text(ddl))
                    conn.commit()
                    logger.info("Migrated: added column %s.%s", table.name, col.name)
                except Exception as e:
                    logger.warning("Migration skipped %s.%s: %s", table.name, col.name, e)


def init_db() -> Engine:
    """Initialize the database: create engine and all tables.

    This is the primary entry point called at application startup.
    Creates all ORM-defined tables if they don't already exist, and
    applies any missing column migrations for SQLite.

    Returns:
        Engine: The initialized database engine.

    Example:
        >>> engine = init_db()
        >>> # All tables now exist in the connected database
    """
    engine = create_engine_with_fallback()
    Base.metadata.create_all(engine)
    _migrate_sqlite_columns(engine)
    logger.info(
        "Database tables created/verified (backend=%s)", ACTIVE_DB_BACKEND
    )
    return engine


def get_engine() -> Engine:
    """Get the current database engine, initializing if needed.

    Returns:
        Engine: The active database engine.

    Example:
        >>> engine = get_engine()
    """
    global _engine
    if _engine is None:
        init_db()
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session that auto-closes on exit.

    Yields a SQLAlchemy session within a context manager. Commits on
    successful completion, rolls back on exception, and always closes
    the session.

    Yields:
        Session: SQLAlchemy session for database operations.

    Example:
        >>> with get_session() as session:
        ...     trades = session.query(Trade).filter_by(status="OPEN").all()
    """
    global _SessionFactory
    if _SessionFactory is None:
        init_db()

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
