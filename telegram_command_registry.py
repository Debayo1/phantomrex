"""Central command metadata and capability policy for Telegram handlers."""

from dataclasses import dataclass
from typing import Optional, Tuple


ROLE_VIEWER = "viewer"
ROLE_ADMIN = "admin"
ROLE_GOD = "god"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    category: str
    capability: str
    minimum_role: str
    aliases: Tuple[str, ...] = ()
    destructive: bool = False
    confirmation_required: bool = False


READ_ONLY = "read_only"
ADVISORY = "advisory"
TRADING_CONTROL = "trading_control"
OWNER_CONFIGURATION = "owner_configuration"
SYSTEM_LIFECYCLE = "system_lifecycle"
ADMINISTRATION = "administration"


MENU_CATEGORIES = (
    ("observe", "Observe", (
        "status", "dbstatus", "balance", "positions", "stats", "spstats", "news", "account",
    ), READ_ONLY),
    ("understand", "Understand", (
        "why", "suggest", "ask", "analyze", "explain",
    ), ADVISORY),
    ("trading", "Trading Control", (
        "pause", "resume", "protect", "closeall",
    ), TRADING_CONTROL),
    ("risk", "Risk and Account", (
        "settings", "login", "logout", "saveaccount",
    ), OWNER_CONFIGURATION),
    ("models", "Models and System", (
        "retrain", "rollback", "restart", "update", "resetguard", "tokens", "setgemini",
    ), SYSTEM_LIFECYCLE),
    ("administration", "Administration", (
        "adduser", "grant", "removeuser", "listusers",
    ), ADMINISTRATION),
)


COMMANDS = {
    "menu": CommandSpec("menu", "Open categorized control menu", "observe", READ_ONLY, ROLE_VIEWER, ("start", "help")),
    "status": CommandSpec("status", "Live market telemetry", "observe", READ_ONLY, ROLE_VIEWER),
    "dbstatus": CommandSpec("dbstatus", "Database path, integrity, and ledger counts", "observe", READ_ONLY, ROLE_VIEWER),
    "balance": CommandSpec("balance", "Account balance and PnL", "observe", READ_ONLY, ROLE_VIEWER),
    "positions": CommandSpec("positions", "List active positions", "observe", READ_ONLY, ROLE_VIEWER),
    "account": CommandSpec("account", "View active MT5 account", "observe", READ_ONLY, ROLE_VIEWER),
    "stats": CommandSpec("stats", "Performance and R-multiples", "observe", READ_ONLY, ROLE_VIEWER, ("performance",)),
    "spstats": CommandSpec("spstats", "Small-profit demo history", "observe", READ_ONLY, ROLE_VIEWER),
    "news": CommandSpec("news", "Upcoming high-impact news", "observe", READ_ONLY, ROLE_VIEWER),

    "why": CommandSpec("why", "Explain the latest decision", "understand", ADVISORY, ROLE_VIEWER),
    "suggest": CommandSpec("suggest", "Evidence-backed advisory next step", "understand", ADVISORY, ROLE_VIEWER),
    "ask": CommandSpec("ask", "Read-only AI explanation", "understand", ADVISORY, ROLE_VIEWER),
    "analyze": CommandSpec("analyze", "AI setup evaluation", "understand", ADVISORY, ROLE_VIEWER),
    "explain": CommandSpec("explain", "Replay one recorded decision", "understand", ADVISORY, ROLE_VIEWER),

    "pause": CommandSpec("pause", "Pause opening new trades", "trading", TRADING_CONTROL, ROLE_ADMIN),
    "resume": CommandSpec("resume", "Resume the trading engine", "trading", TRADING_CONTROL, ROLE_ADMIN),
    "protect": CommandSpec("protect", "Authorize open-position protection", "trading", TRADING_CONTROL, ROLE_ADMIN),
    "closeall": CommandSpec("closeall", "Emergency close all trades", "trading", TRADING_CONTROL, ROLE_GOD, destructive=True, confirmation_required=True),

    "settings": CommandSpec("settings", "View and configure risk settings", "risk", OWNER_CONFIGURATION, ROLE_ADMIN),
    "login": CommandSpec("login", "Switch MT5 account", "risk", OWNER_CONFIGURATION, ROLE_GOD),
    "logout": CommandSpec("logout", "Disconnect MT5 and pause trading", "risk", OWNER_CONFIGURATION, ROLE_GOD, destructive=True, confirmation_required=True),
    "saveaccount": CommandSpec("saveaccount", "Save an MT5 account profile", "risk", OWNER_CONFIGURATION, ROLE_GOD),
    "setrisk": CommandSpec("setrisk", "Set account risk percentage", "risk", OWNER_CONFIGURATION, ROLE_ADMIN),
    "setthreshold": CommandSpec("setthreshold", "Set AI confidence threshold", "risk", OWNER_CONFIGURATION, ROLE_ADMIN),
    "setmaxpos": CommandSpec("setmaxpos", "Set maximum positions", "risk", OWNER_CONFIGURATION, ROLE_ADMIN),
    "togglebreakeven": CommandSpec("togglebreakeven", "Toggle breakeven management", "risk", OWNER_CONFIGURATION, ROLE_ADMIN),
    "toggletrailing": CommandSpec("toggletrailing", "Toggle trailing stops", "risk", OWNER_CONFIGURATION, ROLE_ADMIN),

    "retrain": CommandSpec("retrain", "Train candidate AI models", "models", SYSTEM_LIFECYCLE, ROLE_GOD),
    "rollback": CommandSpec("rollback", "Restore previous AI model", "models", SYSTEM_LIFECYCLE, ROLE_GOD, destructive=True, confirmation_required=True),
    "restart": CommandSpec("restart", "Restart the bot engine", "models", SYSTEM_LIFECYCLE, ROLE_GOD, destructive=True, confirmation_required=True),
    "update": CommandSpec("update", "Pull latest approved code", "models", SYSTEM_LIFECYCLE, ROLE_GOD, destructive=True, confirmation_required=True),
    "resetguard": CommandSpec("resetguard", "Reset daily-loss guard state", "models", SYSTEM_LIFECYCLE, ROLE_GOD, destructive=True, confirmation_required=True),
    "tokens": CommandSpec("tokens", "Gemini usage and quota", "models", SYSTEM_LIFECYCLE, ROLE_ADMIN, ("usage",)),
    "setgemini": CommandSpec("setgemini", "Change Gemini API key", "models", SYSTEM_LIFECYCLE, ROLE_GOD, ("setkey",), destructive=True),

    "adduser": CommandSpec("adduser", "Authorize a Telegram user", "administration", ADMINISTRATION, ROLE_GOD),
    "grant": CommandSpec("grant", "Change a user's role", "administration", ADMINISTRATION, ROLE_GOD, destructive=True, confirmation_required=True),
    "removeuser": CommandSpec("removeuser", "Remove Telegram authorization", "administration", ADMINISTRATION, ROLE_GOD, destructive=True, confirmation_required=True),
    "listusers": CommandSpec("listusers", "List authorized users", "administration", ADMINISTRATION, ROLE_ADMIN),

    "signal": CommandSpec("signal", "Review a pasted trade signal", "trading", TRADING_CONTROL, ROLE_ADMIN),
    "buy": CommandSpec("buy", "Quick market buy", "trading", TRADING_CONTROL, ROLE_ADMIN, destructive=True, confirmation_required=True),
    "sell": CommandSpec("sell", "Quick market sell", "trading", TRADING_CONTROL, ROLE_ADMIN, destructive=True, confirmation_required=True),
}


ROLE_RANK = {ROLE_VIEWER: 0, ROLE_ADMIN: 1, ROLE_GOD: 2}


def get_command(name: str) -> Optional[CommandSpec]:
    normalized = str(name).lower().lstrip("/")
    direct = COMMANDS.get(normalized)
    if direct:
        return direct
    for spec in COMMANDS.values():
        if normalized in spec.aliases:
            return spec
    return None


def role_allowed(user_role: str, spec: CommandSpec) -> bool:
    if user_role not in ROLE_RANK or spec.minimum_role not in ROLE_RANK:
        return False
    return ROLE_RANK[user_role] >= ROLE_RANK[spec.minimum_role]


def category_commands(category: str):
    for spec in COMMANDS.values():
        if spec.category == category:
            yield spec
