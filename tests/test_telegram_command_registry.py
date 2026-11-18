from telegram_command_registry import (
    COMMANDS,
    ROLE_ADMIN,
    ROLE_GOD,
    ROLE_VIEWER,
    get_command,
    role_allowed,
)


def test_legacy_aliases_resolve_to_the_primary_command():
    assert get_command("performance").name == "stats"
    assert get_command("usage").name == "tokens"
    assert get_command("start").name == "menu"


def test_role_policy_is_hierarchical():
    status = get_command("status")
    retrain = get_command("retrain")
    closeall = get_command("closeall")

    assert role_allowed(ROLE_VIEWER, status)
    assert not role_allowed(ROLE_VIEWER, retrain)
    assert role_allowed(ROLE_GOD, retrain)
    assert role_allowed(ROLE_GOD, closeall)


def test_destructive_commands_require_confirmation():
    assert get_command("closeall").destructive
    assert get_command("closeall").confirmation_required
    assert get_command("rollback").destructive


def test_advisory_commands_do_not_gain_trading_capability():
    for name in ("why", "suggest", "ask", "analyze"):
        spec = get_command(name)
        assert spec.category == "understand"
        assert spec.capability in ("advisory", "read_only")


def test_registry_contains_no_duplicate_command_names():
    assert len(COMMANDS) == len(set(COMMANDS.keys()))
