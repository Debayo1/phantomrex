"""Shared persistent database path configuration and diagnostics."""

import os
import sqlite3
from pathlib import Path
from urllib.parse import quote


DATABASE_FILENAME = "signals.db"
DATA_DIR_ENV = "PHANTOMREX_DATA_DIR"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def resolve_db_path(data_dir=None):
    """Resolve the shared SQLite file, preserving the existing default location."""
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    directory = Path(data_dir).expanduser().resolve()
    return str(directory / DATABASE_FILENAME)


def ensure_db_parent(db_path):
    """Create the configured parent directory or raise a path-specific error."""
    path = Path(db_path).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create database directory '{path.parent}': {exc}"
        ) from exc
    return str(path)


def _readonly_uri(db_path):
    path = Path(db_path).resolve()
    return f"file:{quote(path.as_posix(), safe='/:')}?mode=ro"


def initialize_database(db_path=None):
    """Initialize both ledgers at the single configured path and report readiness."""
    path = str(Path(db_path or resolve_db_path()).expanduser().resolve())
    try:
        import db_ledger
        import small_profit_ledger

        db_ledger.init_db(path)
        small_profit_ledger.init_db(path)
    except (OSError, sqlite3.Error) as exc:
        return {
            "path": path,
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return get_database_status(path)


def get_database_status(db_path=None):
    """Return read-only database and filesystem diagnostics without creating files."""
    path = Path(db_path or resolve_db_path()).expanduser().resolve()
    parent = path.parent
    try:
        exists = path.is_file()
        file_stat = path.stat() if exists else None
        metadata_error = None
    except OSError as exc:
        exists = False
        file_stat = None
        metadata_error = f"{type(exc).__name__}: {exc}"
    directory_exists = parent.is_dir()
    status = {
        "path": str(path),
        "exists": exists,
        "size_bytes": file_stat.st_size if file_stat else None,
        "last_modified": file_stat.st_mtime if file_stat else None,
        "directory_exists": directory_exists,
        "directory_writable": os.access(parent, os.W_OK) if directory_exists else False,
        "file_writable": os.access(path, os.W_OK) if exists else None,
        "integrity": "not_checked",
        "tables": [],
        "row_counts": {},
        "error": metadata_error,
        "ready": False,
    }
    if metadata_error:
        return status
    if not status["exists"]:
        status["error"] = "Database file does not exist at the configured path."
        status["ready"] = False
        return status

    try:
        connection = sqlite3.connect(_readonly_uri(path), uri=True, timeout=5)
        try:
            status["integrity"] = connection.execute("PRAGMA integrity_check;").fetchone()[0]
            tables = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;"
                )
            ]
            status["tables"] = tables
            for table in ("signals", "small_profit_history"):
                if table in tables:
                    escaped = table.replace('"', '""')
                    status["row_counts"][table] = connection.execute(
                        f'SELECT COUNT(*) FROM "{escaped}";'
                    ).fetchone()[0]
                else:
                    status["row_counts"][table] = None
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    required_tables = {"signals", "small_profit_history"}
    status["ready"] = (
        status["error"] is None
        and status["integrity"] == "ok"
        and required_tables.issubset(status["tables"])
        and status["directory_writable"]
        and status["file_writable"]
    )
    if status["error"] is None and not required_tables.issubset(status["tables"]):
        status["error"] = "One or more required ledger tables are missing."
    elif status["error"] is None and not status["ready"]:
        status["error"] = "Database path is not writable by the bot process."
    return status
