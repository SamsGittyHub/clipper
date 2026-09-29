"""SQLite engine and session helpers.

SQLite is the v1 store (PRD 11); swapping `DATABASE_URL` to PostgreSQL is the
only change needed to move up, so no SQLite-specific SQL appears anywhere else.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

log = logging.getLogger("clipz.db")

_connect_args = {}
if settings.database_url.startswith("sqlite"):
    # The pipeline worker runs in its own thread and shares the engine.
    _connect_args = {"check_same_thread": False, "timeout": 30}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - trivial
        cursor = dbapi_connection.cursor()
        # WAL lets the API read job progress while the worker writes it.
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        # Wait instead of failing when several threads write at once (the
        # parallel renderer and the API both update the same job row).
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    from . import config as config_module
    from .models import Base  # imported here so models register their tables first

    Base.metadata.create_all(engine)
    add_missing_columns(Base.metadata, engine)
    # A storage directory adopted from the app's previous name still has the
    # old location baked into every path the database recorded.
    if config_module.adopted_from is not None:
        relocate_stored_paths(config_module.adopted_from, settings.storage_dir)
        config_module.adopted_from = None


def relocate_stored_paths(old_root: Path, new_root: Path) -> int:
    """Repoint absolute paths recorded under a previous storage directory.

    Clips, downloads and transcripts are stored as absolute paths, so moving
    the directory would otherwise leave every preview, download and thumbnail
    pointing at somewhere that no longer exists. Returns how many were fixed.
    """
    from .models import Clip, Video

    targets = (
        (Clip, ("output_path", "subtitle_path")),
        (Video, ("local_path", "audio_path", "transcript_path")),
    )
    prefix = str(old_root)
    replacement = str(new_root)
    changed = 0

    with session_scope() as session:
        for model, fields in targets:
            for row in session.scalars(select(model)):
                for field in fields:
                    value = getattr(row, field, None)
                    if value and value.startswith(prefix):
                        setattr(row, field, replacement + value[len(prefix):])
                        changed += 1

    if changed:
        log.info("Repointed %d stored path(s) from %s to %s", changed, old_root, new_root)
    return changed


def add_missing_columns(metadata, target_engine) -> list[str]:
    """Bring an existing database up to the current model.

    `create_all` only creates tables that do not exist; it never alters one.
    Adding a column to a model would otherwise break every install that
    already has data. Additive changes (new nullable/defaulted columns) are
    the only kind this project makes, and `ALTER TABLE ... ADD COLUMN` is
    standard SQL, so this covers them without a migration framework.
    """
    applied: list[str] = []
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())
    with target_engine.begin() as connection:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                column_type = column.type.compile(dialect=target_engine.dialect)
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'
                default = column.default.arg if column.default is not None else None
                if isinstance(default, bool):
                    ddl += f" DEFAULT {1 if default else 0}"
                elif isinstance(default, (int, float)):
                    ddl += f" DEFAULT {default}"
                elif isinstance(default, str):
                    ddl += f" DEFAULT '{default}'"
                log.info("Migrating: %s", ddl)
                connection.execute(text(ddl))
                applied.append(f"{table.name}.{column.name}")
    return applied


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for worker code."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
