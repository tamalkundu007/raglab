"""
Async SQLAlchemy engine, declarative base, and session factory for RAGLab.

All services that need Postgres import from here. The DSN is loaded from
RAGLAB_POSTGRES_DSN via BaseServiceSettings — never hardcoded.

Usage:
    from raglab_common.db import Base, get_session, init_db

    # In service startup (lifespan):
    await init_db(dsn)

    # In a request handler:
    async with get_session() as session:
        session.add(...)
        await session.commit()
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from raglab_common.logging import get_logger

log = get_logger(__name__)

# Naming convention for auto-generated constraint names (alembic-friendly)
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """Declarative base for all RAGLab ORM models."""
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


async def init_db(dsn: str, echo: bool = False) -> None:
    """
    Initialise the async engine and session factory.

    Call once at service startup (inside lifespan context manager).

    Args:
        dsn:  PostgreSQL DSN — postgresql+asyncpg://user:pass@host/db
        echo: Log all SQL statements (dev only).
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        dsn,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    log.info("db.engine_created", dsn=dsn.split("@")[-1])  # log host/db only


async def create_tables() -> None:
    """Create all tables defined in Base.metadata (dev/test only)."""
    global _engine
    if _engine is None:
        raise RuntimeError("Call init_db() before create_tables()")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db.tables_created")


async def close_db() -> None:
    """Dispose engine on service shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        log.info("db.engine_closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager yielding a database session.

    Automatically commits on clean exit, rolls back on exception.

    Usage:
        async with get_session() as session:
            session.add(model_instance)
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() at startup.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
