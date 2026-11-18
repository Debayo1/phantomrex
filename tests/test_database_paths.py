import os
import sqlite3

from db_paths import get_database_status, initialize_database, resolve_db_path
import db_ledger
import small_profit_ledger


def test_ledgers_share_configured_database_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PHANTOMREX_DATA_DIR", str(tmp_path / "persistent" / "state"))
    expected = os.path.join(str(tmp_path / "persistent" / "state"), "signals.db")

    assert resolve_db_path() == expected
    assert small_profit_ledger.resolve_db_path() == expected
    assert db_ledger.resolve_db_path() == expected


def test_ledger_creates_missing_database_parent(tmp_path):
    database = tmp_path / "nested" / "persistent" / "signals.db"

    status = initialize_database(str(database))

    assert database.is_file()
    assert status["ready"] is True
    assert status["integrity"] == "ok"
    assert status["row_counts"]["signals"] == 0
    assert status["row_counts"]["small_profit_history"] == 0


def test_read_only_status_does_not_create_missing_database(tmp_path):
    database = tmp_path / "not-created" / "signals.db"

    status = get_database_status(str(database))

    assert not database.exists()
    assert status["exists"] is False
    assert status["integrity"] == "not_checked"
    assert "does not exist" in status["error"]


def test_database_status_reports_missing_ledger_schema(tmp_path):
    database = tmp_path / "schema-missing.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY);")

    status = get_database_status(str(database))

    assert status["ready"] is False
    assert "required ledger tables are missing" in status["error"]


def test_database_status_reports_malformed_database_without_creating_replacement(tmp_path):
    database = tmp_path / "malformed.db"
    database.write_bytes(b"not a sqlite database")

    status = get_database_status(str(database))

    assert status["exists"] is True
    assert status["integrity"] == "not_checked"
    assert "DatabaseError" in status["error"] or "file is not a database" in status["error"]
    assert database.read_bytes() == b"not a sqlite database"


def test_database_path_override_is_not_silently_replaced(tmp_path):
    unavailable = tmp_path / "file-as-parent"
    unavailable.write_text("occupied", encoding="utf-8")
    database = unavailable / "signals.db"

    try:
        small_profit_ledger.init_db(str(database))
    except OSError as exc:
        assert "Cannot create database directory" in str(exc)
    else:
        raise AssertionError("Expected a path-specific database setup error")
