import os
import json
import tempfile
import pytest
from telegram_manager import (
    load_config,
    save_config,
    validate_credentials,
    load_env_file,
    is_placeholder,
    DEFAULT_CONFIG,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure environment variables do not bleed across tests."""
    keys_to_clear = [
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "MT5_DEMO_LOGIN",
        "MT5_DEMO_PASSWORD",
        "MT5_DEMO_SERVER",
        "MT5_REAL_LOGIN",
        "MT5_REAL_PASSWORD",
        "MT5_REAL_SERVER",
        "MT5_LIVE_LOGIN",
        "MT5_LIVE_PASSWORD",
        "MT5_LIVE_SERVER",
        "MT5_ACTIVE_ACCOUNT",
    ]
    for k in keys_to_clear:
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


def test_is_placeholder():
    assert is_placeholder("") is True
    assert is_placeholder(None) is True
    assert is_placeholder("0") is True
    assert is_placeholder("YOUR_TELEGRAM_BOT_TOKEN_HERE") is True
    assert is_placeholder("YOUR_DEMO_PASSWORD") is True
    assert is_placeholder("PLACEHOLDER") is True
    assert is_placeholder("valid_token_12345") is False
    assert is_placeholder("StrongPassword!987") is False


def test_load_config_defaults(clean_env, tmp_path):
    dummy_path = str(tmp_path / "nonexistent_config.json")
    cfg = load_config(config_path=dummy_path)
    assert "settings" in cfg
    assert "accounts" in cfg
    assert "demo" in cfg["accounts"]
    assert "live" in cfg["accounts"]
    assert cfg["bot_token"] == ""


def test_load_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "# Comment line\n"
        "TELEGRAM_BOT_TOKEN='token_from_env_file'\n"
        "MT5_DEMO_LOGIN=\"1234567\"\n"
        "MT5_DEMO_PASSWORD=pass123\n",
        encoding="utf-8"
    )
    loaded = load_env_file(str(env_file))
    assert loaded is True
    assert os.environ.get("TELEGRAM_BOT_TOKEN") == "token_from_env_file"
    assert os.environ.get("MT5_DEMO_LOGIN") == "1234567"
    assert os.environ.get("MT5_DEMO_PASSWORD") == "pass123"


def test_env_overrides_demo_and_real(clean_env, tmp_path):
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC-DEF")
    clean_env.setenv("GEMINI_API_KEY", "gemini_secret_key")
    clean_env.setenv("MT5_DEMO_LOGIN", "477157874")
    clean_env.setenv("MT5_DEMO_PASSWORD", "demo_secret_pass")
    clean_env.setenv("MT5_DEMO_SERVER", "Exness-MT5Trial9")
    clean_env.setenv("MT5_REAL_LOGIN", "295802448")
    clean_env.setenv("MT5_REAL_PASSWORD", "real_secret_pass")
    clean_env.setenv("MT5_REAL_SERVER", "Exness-MT5Real27")
    clean_env.setenv("MT5_ACTIVE_ACCOUNT", "live")

    cfg_file = str(tmp_path / "config.json")
    cfg = load_config(config_path=cfg_file)

    assert cfg["bot_token"] == "123456:ABC-DEF"
    assert cfg["gemini_api_key"] == "gemini_secret_key"
    assert cfg["accounts"]["demo"]["login"] == 477157874
    assert cfg["accounts"]["demo"]["password"] == "demo_secret_pass"
    assert cfg["accounts"]["demo"]["server"] == "Exness-MT5Trial9"
    assert cfg["accounts"]["live"]["login"] == 295802448
    assert cfg["accounts"]["live"]["password"] == "real_secret_pass"
    assert cfg["accounts"]["live"]["server"] == "Exness-MT5Real27"
    assert cfg["active_account"] == "live"


def test_save_config_never_persists_env_secrets(clean_env, tmp_path):
    """Addendum D1: save_config must NEVER persist secrets resolved from environment variables."""
    cfg_file = str(tmp_path / "config.json")
    
    # Pre-populate disk file with non-secret settings and placeholders
    initial_disk = {
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN_HERE",
        "accounts": {
            "demo": {
                "login": 0,
                "password": "YOUR_DEMO_PASSWORD",
                "server": "Exness-MT5Trial9"
            }
        },
        "settings": {
            "risk_percent": 1.5
        }
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(initial_disk, f)

    # Set secret in environment
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "real_live_telegram_token")
    clean_env.setenv("MT5_DEMO_PASSWORD", "real_live_demo_password")

    # Load configuration
    cfg = load_config(config_path=cfg_file)
    assert cfg["bot_token"] == "real_live_telegram_token"
    assert cfg["accounts"]["demo"]["password"] == "real_live_demo_password"

    # Modify non-secret runtime setting and save
    cfg["settings"]["risk_percent"] = 2.0
    cfg["is_paused"] = True
    save_config(cfg, config_path=cfg_file)

    # Read the file on disk directly
    with open(cfg_file, "r", encoding="utf-8") as f:
        saved_on_disk = json.load(f)

    # SECRETS MUST NOT BE SAVED TO DISK
    assert saved_on_disk["bot_token"] != "real_live_telegram_token"
    assert saved_on_disk["accounts"]["demo"]["password"] != "real_live_demo_password"
    # Placeholder or empty string from disk must be preserved
    assert saved_on_disk["bot_token"] == "YOUR_TELEGRAM_BOT_TOKEN_HERE"
    assert saved_on_disk["accounts"]["demo"]["password"] == "YOUR_DEMO_PASSWORD"

    # Non-secret settings MUST be persisted
    assert saved_on_disk["settings"]["risk_percent"] == 2.0
    assert saved_on_disk["is_paused"] is True


def test_validate_credentials_rejection():
    # Placeholder bot token
    cfg_placeholder = {
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN_HERE",
        "active_account": "demo",
        "accounts": {
            "demo": {"login": 12345, "password": "valid_password", "server": "Exness-Server"}
        }
    }
    valid, reason = validate_credentials(cfg_placeholder)
    assert valid is False
    assert "token" in reason.lower()

    # Placeholder password
    cfg_placeholder_pw = {
        "bot_token": "12345:ValidToken",
        "active_account": "demo",
        "accounts": {
            "demo": {"login": 12345, "password": "YOUR_DEMO_PASSWORD", "server": "Exness-Server"}
        }
    }
    valid, reason = validate_credentials(cfg_placeholder_pw)
    assert valid is False
    assert "password" in reason.lower()

    # Zero login
    cfg_zero_login = {
        "bot_token": "12345:ValidToken",
        "active_account": "demo",
        "accounts": {
            "demo": {"login": 0, "password": "valid_password", "server": "Exness-Server"}
        }
    }
    valid, reason = validate_credentials(cfg_zero_login)
    assert valid is False
    assert "login" in reason.lower()


def test_validate_credentials_valid():
    cfg_valid = {
        "bot_token": "123456:ABC-DEF1234ghijk",
        "active_account": "demo",
        "accounts": {
            "demo": {"login": 477157874, "password": "SecretPassword!123", "server": "Exness-MT5Trial9"}
        }
    }
    valid, reason = validate_credentials(cfg_valid)
    assert valid is True
    assert reason == "OK"
