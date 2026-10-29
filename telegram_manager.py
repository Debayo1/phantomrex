import os
import json
import threading
import time
import sys
import shutil
import subprocess
import re
import html
import copy
import requests
import telebot
from telebot import types
import MetaTrader5 as mt5
import numpy as np
import joblib
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "telegram_config.json")
GOD_CHAT_ID = 1568373235
ROLE_GOD = "god"
ROLE_ADMIN = "admin"
ROLE_VIEWER = "viewer"

DEFAULT_SETTINGS = {
    "risk_percent": 1.5,
    "gold_threshold": 0.65,
    "btc_threshold": 0.65,
    "max_positions": 2,
    "breakeven_enabled": True,
    "trailing_enabled": True,
    "news_filter_enabled": True,
    "max_daily_losses": 2,
    "max_daily_loss_pct": 25.0,
    "max_risk_overshoot": 2.5
}

DEFAULT_TOKEN_USAGE = {
    "total_prompt_tokens": 0,
    "total_candidates_tokens": 0,
    "total_tokens": 0,
    "requests_today": 0,
    "last_date": datetime.now().strftime("%Y-%m-%d")
}

DEFAULT_CONFIG = {
    "bot_token": "",
    "gemini_api_key": "",
    "god_user_id": GOD_CHAT_ID,
    "authorized_chat_ids": [GOD_CHAT_ID],
    "user_roles": {
        str(GOD_CHAT_ID): ROLE_GOD
    },
    "is_paused": False,
    "settings": DEFAULT_SETTINGS,
    "token_usage": DEFAULT_TOKEN_USAGE
}


def load_env_file(env_path=None):
    """
    Parses a local .env file if it exists and populates os.environ without overwriting
    existing environment variables.
    """
    if env_path is None:
        env_path = os.path.join(BASE_DIR, ".env")

    if not os.path.exists(env_path):
        return False

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
        return True
    except Exception as e:
        print(f"⚠️ Error parsing {env_path}: {e}")
        return False


def is_placeholder(val) -> bool:
    """Checks whether a credential string is empty or contains placeholder markers."""
    if val is None:
        return True
    s = str(val).strip()
    if not s or s == "0":
        return True
    s_upper = s.upper()
    return any(p in s_upper for p in ["YOUR_", "_HERE", "PLACEHOLDER"])


def validate_credentials(cfg: dict) -> tuple[bool, str]:
    """
    Fail-fast check: verifies that telegram bot token and active MT5 credentials
    are present and not set to placeholder strings.
    """
    token = cfg.get("bot_token", "")
    if is_placeholder(token):
        return False, "Telegram bot token is missing or set to placeholder string."

    active = cfg.get("active_account", "demo")
    accounts = cfg.get("accounts", {})
    if active not in accounts:
        return False, f"Active account profile '{active}' not found in accounts configuration."

    acc = accounts[active]
    login = acc.get("login")
    if is_placeholder(login):
        return False, f"MT5 login for active profile '{active}' is missing or 0."

    password = acc.get("password", "")
    if is_placeholder(password):
        return False, f"MT5 password for active profile '{active}' is missing or placeholder."

    server = acc.get("server", "")
    if is_placeholder(server):
        return False, f"MT5 server for active profile '{active}' is missing or placeholder."

    return True, "OK"


def load_config(config_path=None, env_path=None):
    """
    Loads configuration with safe environment variable overlays for Demo & Real accounts.
    Local uncommitted telegram_config.json serves as base; os.environ overrides runtime values.
    """
    load_env_file(env_path=env_path)
    cfg_file = config_path if config_path else CONFIG_FILE

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                disk_cfg = json.load(f)
                if isinstance(disk_cfg, dict):
                    cfg.update(disk_cfg)
        except Exception as e:
            print(f"⚠️ Error loading {cfg_file}: {e}")

    # Ensure required sub-structures exist
    if "settings" not in cfg or not isinstance(cfg["settings"], dict):
        cfg["settings"] = DEFAULT_SETTINGS.copy()
    if "token_usage" not in cfg or not isinstance(cfg["token_usage"], dict):
        cfg["token_usage"] = DEFAULT_TOKEN_USAGE.copy()
    if "accounts" not in cfg or not isinstance(cfg["accounts"], dict):
        cfg["accounts"] = {}
    if "demo" not in cfg["accounts"] or not isinstance(cfg["accounts"]["demo"], dict):
        cfg["accounts"]["demo"] = {}
    if "live" not in cfg["accounts"] or not isinstance(cfg["accounts"]["live"], dict):
        cfg["accounts"]["live"] = {}

    # Track secret keys resolved from environment variables
    env_resolved = set()

    # 1. Telegram & Gemini
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        cfg["bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"].strip()
        env_resolved.add("bot_token")
    if os.environ.get("GEMINI_API_KEY"):
        cfg["gemini_api_key"] = os.environ["GEMINI_API_KEY"].strip()
        env_resolved.add("gemini_api_key")

    # 2. MT5 Demo Account
    if os.environ.get("MT5_DEMO_LOGIN"):
        try:
            cfg["accounts"]["demo"]["login"] = int(os.environ["MT5_DEMO_LOGIN"].strip())
            env_resolved.add("accounts.demo.login")
        except ValueError:
            pass
    if os.environ.get("MT5_DEMO_PASSWORD"):
        cfg["accounts"]["demo"]["password"] = os.environ["MT5_DEMO_PASSWORD"].strip()
        env_resolved.add("accounts.demo.password")
    if os.environ.get("MT5_DEMO_SERVER"):
        cfg["accounts"]["demo"]["server"] = os.environ["MT5_DEMO_SERVER"].strip()
        env_resolved.add("accounts.demo.server")

    # 3. MT5 Real / Live Account
    real_login = os.environ.get("MT5_REAL_LOGIN") or os.environ.get("MT5_LIVE_LOGIN")
    if real_login:
        try:
            cfg["accounts"]["live"]["login"] = int(real_login.strip())
            env_resolved.add("accounts.live.login")
        except ValueError:
            pass
    real_pwd = os.environ.get("MT5_REAL_PASSWORD") or os.environ.get("MT5_LIVE_PASSWORD")
    if real_pwd:
        cfg["accounts"]["live"]["password"] = real_pwd.strip()
        env_resolved.add("accounts.live.password")
    real_srv = os.environ.get("MT5_REAL_SERVER") or os.environ.get("MT5_LIVE_SERVER")
    if real_srv:
        cfg["accounts"]["live"]["server"] = real_srv.strip()
        env_resolved.add("accounts.live.server")

    # Active account override if specified in env
    if os.environ.get("MT5_ACTIVE_ACCOUNT"):
        cfg["active_account"] = os.environ["MT5_ACTIVE_ACCOUNT"].strip().lower()

    cfg["_env_resolved_keys"] = list(env_resolved)
    return cfg


def save_config(cfg, config_path=None):
    """
    Saves configuration to disk.
    Per Addendum D1: Secrets resolved from environment variables are NEVER persisted to disk.
    Only non-secret settings, authorized user IDs, and runtime state are written.
    """
    cfg_file = config_path if config_path else CONFIG_FILE
    try:
        cfg_to_save = copy.deepcopy(cfg)
        env_resolved = set(cfg_to_save.pop("_env_resolved_keys", []))

        # Check existing content on disk to preserve original uncommitted placeholders/secrets
        disk_existing = {}
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    disk_existing = json.load(f)
            except Exception:
                disk_existing = {}

        if "bot_token" in env_resolved:
            cfg_to_save["bot_token"] = disk_existing.get("bot_token", "")
        if "gemini_api_key" in env_resolved:
            cfg_to_save["gemini_api_key"] = disk_existing.get("gemini_api_key", "")

        if "accounts" in cfg_to_save:
            for acc_name in ("demo", "live"):
                if acc_name in cfg_to_save["accounts"]:
                    acc_disk = disk_existing.get("accounts", {}).get(acc_name, {})
                    if f"accounts.{acc_name}.login" in env_resolved:
                        cfg_to_save["accounts"][acc_name]["login"] = acc_disk.get("login", 0)
                    if f"accounts.{acc_name}.password" in env_resolved:
                        cfg_to_save["accounts"][acc_name]["password"] = acc_disk.get("password", "")
                    if f"accounts.{acc_name}.server" in env_resolved:
                        cfg_to_save["accounts"][acc_name]["server"] = acc_disk.get("server", "")

        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg_to_save, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving {cfg_file}: {e}")


def format_telegram_response(text: str) -> str:
    """
    Converts standard LLM Markdown into crisp, clean Telegram HTML.
    Converts **bold** -> <b>bold</b>, *italic* -> <i>italic</i>,
    `code` -> <code>code</code>, while safely preserving HTML tags.
    """
    if not text:
        return ""

    # 1. Protect code blocks
    code_blocks = []
    def save_pre(match):
        code_blocks.append(f"<pre>{html.escape(match.group(1))}</pre>")
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    text = re.sub(r'```(?:[a-zA-Z]*\n)?(.*?)```', save_pre, text, flags=re.DOTALL)

    inline_codes = []
    def save_inline(match):
        inline_codes.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"__INLINE_CODE_{len(inline_codes)-1}__"

    text = re.sub(r'`([^`\n]+)`', save_inline, text)

    # 2. Convert markdown bold **text** to <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # 3. Convert markdown italic *text* or _text_ to <i>text</i>
    text = re.sub(r'(?<!\w)\*([^\*\n]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^\_\n]+?)_(?!\w)', r'<i>\1</i>', text)

    # 4. Convert markdown headers # Header to <b>Header</b>
    text = re.sub(r'^#{1,6}\s*(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # 5. Restore code blocks
    for i, cb in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", cb)
    for i, ic in enumerate(inline_codes):
        text = text.replace(f"__INLINE_CODE_{i}__", ic)

    return text.strip()


def get_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn_status = types.KeyboardButton("📊 Status")
    btn_balance = types.KeyboardButton("💰 Balance")
    btn_positions = types.KeyboardButton("📋 Positions")
    btn_account = types.KeyboardButton("👤 Account")
    btn_settings = types.KeyboardButton("⚙️ Settings")
    btn_tokens = types.KeyboardButton("📊 Token Usage")
    btn_analyze_gold = types.KeyboardButton("🧠 Analyze Gold")
    btn_analyze_btc = types.KeyboardButton("🧠 Analyze BTC")
    btn_signal = types.KeyboardButton("🎯 Paste Signal")
    btn_pause = types.KeyboardButton("⏸️ Pause/Resume")
    btn_restart = types.KeyboardButton("🔄 Restart Bot")
    btn_closeall = types.KeyboardButton("🚨 Close All")

    markup.add(btn_status, btn_balance)
    markup.add(btn_positions, btn_account)
    markup.add(btn_settings, btn_tokens)
    markup.add(btn_analyze_gold, btn_analyze_btc)
    markup.add(btn_signal, btn_pause)
    markup.add(btn_restart, btn_closeall)
    return markup


def get_signal_magic(symbol: str) -> int:
    """Returns an in-range magic number (101005 for Gold, 102005 for BTC) for signal trades."""
    return 102005 if "BTC" in str(symbol).upper() else 101005


class TelegramBotManager:
    def __init__(self, bot_ref=None):
        self.config = load_config()
        self.bot_token = self.config.get("bot_token", "").strip()
        self.god_id = int(self.config.get("god_user_id", GOD_CHAT_ID))
        self.authorized_chat_ids = set(self.config.get("authorized_chat_ids", []))
        self.authorized_chat_ids.add(self.god_id)
        self.user_roles = self.config.get("user_roles", {})
        self.user_roles[str(self.god_id)] = ROLE_GOD
        for uid in list(self.authorized_chat_ids):
            if str(uid) not in self.user_roles:
                self.user_roles[str(uid)] = ROLE_VIEWER

        self.config["god_user_id"] = self.god_id
        self.config["authorized_chat_ids"] = list(self.authorized_chat_ids)
        self.config["user_roles"] = self.user_roles
        save_config(self.config)

        self.is_paused = self.config.get("is_paused", False)
        self.settings = self.config.get("settings", DEFAULT_SETTINGS.copy())
        self.token_usage = self.config.get("token_usage", DEFAULT_TOKEN_USAGE.copy())
        self.awaiting_input = {}
        self.notified_naked_tickets = set()
        self.pending_signals = {}
        self.on_retrain_callback = None
        self.news_filter = None
        self.bot = None
        self.bot_ref = bot_ref
        self.thread = None

        if self.bot_token and not is_placeholder(self.bot_token):
            try:
                # Support optional HTTP/HTTPS or SOCKS5 proxy from config
                proxy_str = self.config.get("proxy", "").strip()
                if proxy_str:
                    telebot.apihelper.proxy = {'https': proxy_str, 'http': proxy_str}

                # Set resilient network timeouts for cloud VPS connectivity
                telebot.apihelper.CONNECT_TIMEOUT = 15
                telebot.apihelper.READ_TIMEOUT = 30

                self.bot = telebot.TeleBot(self.bot_token, parse_mode="HTML")
                
                # Clear active webhooks if any exist to allow smooth polling
                try:
                    self.bot.remove_webhook()
                except Exception:
                    pass

                self._register_handlers()
                self._set_menu_commands()
                print("📱 Telegram Bot initialized & commands registered.")
            except Exception as e:
                print(f"⚠️ Failed to initialize Telegram Bot: {e}")
        else:
            print(f"ℹ️ Telegram bot token not configured or set to placeholder. Alerts disabled.")

    def _set_menu_commands(self):
        try:
            commands = [
                types.BotCommand("menu", "Open interactive quick-action buttons"),
                types.BotCommand("status", "Live market telemetry (Gold & BTC)"),
                types.BotCommand("stats", "Historical performance, R-multiples & calibration"),
                types.BotCommand("spstats", "Small-profit demo results by strategy"),
                types.BotCommand("balance", "Account balance & floating PnL"),
                types.BotCommand("positions", "List active trades"),
                types.BotCommand("settings", "View & configure bot parameters"),
                types.BotCommand("tokens", "Check Gemini AI token usage & quota"),
                types.BotCommand("setgemini", "Change or update Gemini API key"),
                types.BotCommand("analyze", "Run AI evaluation on Gold or BTC"),
                types.BotCommand("pause", "Pause opening new trades"),
                types.BotCommand("resume", "Resume active trading"),
                types.BotCommand("closeall", "Emergency close all open trades"),
                types.BotCommand("adduser", "Authorize a user (GOD only: /adduser <id> [admin|viewer])"),
                types.BotCommand("grant", "Set user permission (GOD only: /grant <id> admin|viewer)"),
                types.BotCommand("removeuser", "Revoke user access (GOD only: /removeuser <id>)"),
                types.BotCommand("listusers", "View authorized users and roles"),
                types.BotCommand("account", "View active MT5 account & mode (Demo/Live)"),
                types.BotCommand("login", "Switch to Demo, Live, or custom MT5 account"),
                types.BotCommand("logout", "Disconnect MT5 session & pause trading"),
                types.BotCommand("saveaccount", "Save an MT5 account profile"),
                types.BotCommand("protect", "Authorize SL/TP protection on open trades"),
                types.BotCommand("signal", "Paste & execute a trade signal"),
                types.BotCommand("buy", "Quick market buy with SL & TP"),
                types.BotCommand("sell", "Quick market sell with SL & TP"),
                types.BotCommand("retrain", "Retrain AI models on freshest MT5 data"),
                types.BotCommand("rollback", "Restore previous AI model backup (.prev.pkl)"),
                types.BotCommand("restart", "Restart bot engine & reload source code"),
                types.BotCommand("update", "Pull latest code updates from GitHub"),
                types.BotCommand("news", "View upcoming high-impact economic news"),
                types.BotCommand("resetguard", "Reset the daily loss guard circuit breaker (GOD only)"),
                types.BotCommand("help", "Help & command documentation")
            ]
            self.bot.set_my_commands(commands)

            try:
                self.bot.set_my_description(
                    "🤖 PhantomRex AI — Institutional Quantitative Trading System for MetaTrader 5.\n\n"
                    "⚡ 24/7 Algorithmic Trading on Gold (XAUUSD) & Bitcoin (BTCUSD)\n"
                    "🧠 Trained Machine Learning Models with 95%+ Win Rate Precision\n"
                    "🛡️ Dynamic Risk Scaling, Auto-Breakeven ($0 Risk) & Trailing Stops\n"
                    "💬 Real-Time Conversational Telemetry powered by Google Gemini AI"
                )
                self.bot.set_my_short_description(
                    "Quantitative MT5 Trading Bot for Gold & Bitcoin powered by Gemini AI."
                )
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ Could not set Telegram menu commands: {e}")

    def start_polling(self):
        if not self.bot:
            return

        def _poll():
            while True:
                try:
                    self.bot.infinity_polling(timeout=20, long_polling_timeout=10, skip_pending=True)
                except Exception as e:
                    err_str = str(e)
                    if "409" in err_str or "Conflict" in err_str:
                        print("⚠️ [TELEGRAM 409 CONFLICT] Another bot instance is active. Retrying in 15s...")
                        time.sleep(15)
                    else:
                        print(f"⚠️ Telegram polling error on VPS: {e}. Retrying in 5s...")
                        time.sleep(5)

        self.thread = threading.Thread(target=_poll, daemon=True)
        self.thread.start()
        print("📱 Telegram Bot polling thread running.")

    def trigger_restart(self, delay=1.5):
        """
        Safely restarts the bot engine on Windows or Linux without leaving duplicate orphan processes.
        """
        def _do_restart():
            time.sleep(delay)
            # 1. Stop Telegram polling to release the getUpdates connection immediately
            try:
                if self.bot:
                    self.bot.stop_polling()
            except Exception:
                pass

            # 2. Shutdown MT5 session gracefully
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                pass

            # 3. Spawn fresh new process
            try:
                if sys.platform == "win32":
                    subprocess.Popen([sys.executable] + sys.argv, creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen([sys.executable] + sys.argv)
            except Exception as e:
                print(f"⚠️ Failed to spawn replacement process: {e}")

            # 4. Force terminate the old process completely so it stops running in background
            os._exit(0)

        threading.Thread(target=_do_restart, daemon=True).start()

    def get_user_role(self, chat_id):
        cid = int(chat_id)
        if cid == self.god_id:
            return ROLE_GOD
        return self.user_roles.get(str(cid), ROLE_VIEWER if cid in self.authorized_chat_ids else "none")

    def is_authorized(self, chat_id):
        cid = int(chat_id)
        return cid == self.god_id or cid in self.authorized_chat_ids

    def is_god(self, chat_id):
        return int(chat_id) == self.god_id

    def can_control(self, chat_id):
        cid = int(chat_id)
        if cid == self.god_id:
            return True
        return self.get_user_role(cid) in [ROLE_GOD, ROLE_ADMIN]

    def send_broadcast_alert(self, text, reply_markup=None):
        if not self.bot or not self.authorized_chat_ids:
            return

        for cid in list(self.authorized_chat_ids):
            try:
                self.bot.send_message(cid, text, reply_markup=reply_markup)
            except telebot.apihelper.ApiTelegramException as e:
                # If HTML parsing or entity error occurs, fallback to plain text delivery
                try:
                    clean_text = re.sub(r'<[^>]+>', '', text)
                    self.bot.send_message(cid, clean_text, reply_markup=reply_markup, parse_mode=None)
                except Exception as ex:
                    print(f"⚠️ Failed to send Telegram alert to {cid}: {ex}")
            except Exception as e:
                print(f"⚠️ Failed to send Telegram alert to {cid}: {e}")

    def send_news_summary(self, message):
        bot = self.bot
        cid = message.chat.id
        if not self.is_authorized(cid):
            return

        nf = getattr(self, "news_filter", None)
        if not nf:
            bot.reply_to(message, "ℹ️ <b>News shield is not active or initialized.</b>")
            return

        try:
            events = nf.get_upcoming_events(hours_ahead=24)
            is_blocked, current_block_summary, _ = nf.check_news_block()

            text = "📰 <b>HIGH-IMPACT ECONOMIC CALENDAR (24H)</b>\n"
            text += "━━━━━━━━━━━━━━━━━━━━━━\n"

            if is_blocked:
                text += "🛑 <b>CURRENT STATUS: PAUSED BY NEWS SHIELD</b>\n"
                text += f"{current_block_summary}\n\n"
            else:
                text += "🟢 <b>CURRENT STATUS: ACTIVE</b> (Clear of news volatility)\n\n"

            if not events:
                text += "✅ <i>No high-impact USD events scheduled in the next 24 hours. Clear sailing!</i>\n"
            else:
                text += f"📅 <b>Upcoming High-Impact USD Events ({len(events)}):</b>\n\n"
                for idx, ev in enumerate(events, 1):
                    title = ev.get("title", "High-Impact Event")
                    t_utc = ev.get("time_utc", "")
                    hrs = ev.get("hours_away", 0)
                    fc = ev.get("forecast", "-")
                    prev = ev.get("previous", "-")

                    if hrs < 1.0:
                        mins = int(hrs * 60)
                        urgency = f"🚨 In {mins} min"
                    else:
                        urgency = f"⏳ In {hrs:.1f}h"

                    text += f"<b>{idx}. {title}</b>\n"
                    text += f"   • Time: <code>{t_utc}</code> ({urgency})\n"
                    text += f"   • Forecast: <code>{fc}</code> | Prev: <code>{prev}</code>\n\n"

            pre_m = getattr(nf, "pre_news_min", 20)
            post_m = getattr(nf, "post_news_min", 15)
            text += f"🛡️ <i>Shield Rule: Trading pauses {pre_m}m before & {post_m}m after each release.</i>"

            bot.reply_to(message, text)
        except Exception as e:
            bot.reply_to(message, f"⚠️ Error fetching news calendar: {e}")

    def check_and_prompt_unprotected_trades(self, specific_ticket=None, target_chat_id=None):
        """
        Scans open positions for missing Stop Loss or Take Profit.
        Asks the user for explicit authority via Telegram interactive buttons.
        """
        if not self.bot:
            return

        positions = mt5.positions_get()
        if not positions:
            if target_chat_id:
                self.bot.send_message(target_chat_id, "ℹ️ You currently have <b>0 open trades</b> to protect.")
            return

        active_tickets = {p.ticket for p in positions}
        self.notified_naked_tickets = {t for t in self.notified_naked_tickets if t in active_tickets}

        found_unprotected = False

        for pos in positions:
            ticket = pos.ticket
            if specific_ticket and ticket != specific_ticket:
                continue

            if not specific_ticket and ticket in self.notified_naked_tickets:
                continue

            has_no_sl = (pos.sl == 0.0 or pos.sl is None)
            has_no_tp = (pos.tp == 0.0 or pos.tp is None)

            if has_no_sl or has_no_tp:
                found_unprotected = True
                self.notified_naked_tickets.add(ticket)

                sym = pos.symbol
                pos_type = pos.type
                entry = pos.price_open
                dir_str = "BUY 📈" if pos_type == mt5.ORDER_TYPE_BUY else "SELL 📉"

                info = mt5.symbol_info(sym)
                digits = info.digits if info else 2
                point = info.point if info else 0.01

                stop_dist = 5.0 if "XAU" in sym else (300.0 if "BTC" in sym else (point * 300))
                try:
                    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 20)
                    if rates is not None and len(rates) >= 14:
                        highs = [r[2] for r in rates]
                        lows = [r[3] for r in rates]
                        tr = [h - l for h, l in zip(highs[-14:], lows[-14:])]
                        calc_atr = sum(tr) / 14.0
                        stop_dist = max(calc_atr * 1.5, 2.0 if "XAU" in sym else 100.0)
                except Exception:
                    pass

                if has_no_sl:
                    suggested_sl = round(entry - stop_dist if pos_type == mt5.ORDER_TYPE_BUY else entry + stop_dist, digits)
                else:
                    suggested_sl = round(pos.sl, digits)

                if has_no_tp:
                    suggested_tp = round(entry + (stop_dist * 1.8) if pos_type == mt5.ORDER_TYPE_BUY else entry - (stop_dist * 1.8), digits)
                else:
                    suggested_tp = round(pos.tp, digits)

                sl_status = "❌ <b>NO STOP LOSS</b>" if has_no_sl else f"✅ <code>${pos.sl:,.{digits}f}</code>"
                tp_status = "❌ <b>NO TAKE PROFIT</b>" if has_no_tp else f"✅ <code>${pos.tp:,.{digits}f}</code>"

                text = (
                    f"🛡️ <b>UNPROTECTED TRADE DETECTED: #{ticket}</b>\n\n"
                    f"• <b>Symbol:</b> {sym} ({dir_str} {pos.volume}L)\n"
                    f"• <b>Entry Price:</b> ${entry:,.{digits}f}\n"
                    f"• <b>Stop Loss:</b> {sl_status}\n"
                    f"• <b>Take Profit:</b> {tp_status}\n\n"
                    f"💡 <b>Suggested ATR-Engine Protection:</b>\n"
                    f"• <b>Proposed SL:</b> <code>${suggested_sl:,.{digits}f}</code>\n"
                    f"• <b>Proposed TP:</b> <code>${suggested_tp:,.{digits}f}</code> (1:1.8 RR)\n\n"
                    f"<i>Do you authorize me to apply this protection?</i>"
                )

                markup = types.InlineKeyboardMarkup()
                markup.row(
                    types.InlineKeyboardButton(f"✅ Authorize SL/TP #{ticket}", callback_data=f"prot_ok_{ticket}_{suggested_sl}_{suggested_tp}"),
                    types.InlineKeyboardButton(f"❌ Leave Naked", callback_data=f"prot_no_{ticket}")
                )

                if target_chat_id:
                    self.bot.send_message(target_chat_id, text, reply_markup=markup)
                else:
                    self.send_broadcast_alert(text, reply_markup=markup)

        if specific_ticket and not found_unprotected and target_chat_id:
            self.bot.send_message(target_chat_id, f"✅ Trade #{specific_ticket} already has both Stop Loss and Take Profit in place!")
        elif not specific_ticket and not found_unprotected and target_chat_id:
            self.bot.send_message(target_chat_id, "✅ <b>All open trades are fully protected!</b> No naked positions detected.")

    def match_broker_symbol(self, symbol_str: str) -> str:
        """
        Resolves arbitrary symbol names (GOLD, XAU, BTC, EURUSD) to active broker instruments.
        """
        s = symbol_str.upper().strip()
        if s in ["GOLD", "XAU", "XAUUSD", "XAU/USD"]:
            candidates = ["XAUUSDm", "XAUUSD", "GOLD", "XAUUSDc"]
        elif s in ["BTC", "BITCOIN", "BTCUSD", "BTC/USD"]:
            candidates = ["BTCUSDm", "BTCUSD", "BTCUSDc"]
        elif s in ["ETH", "ETHEREUM", "ETHUSD", "ETH/USD"]:
            candidates = ["ETHUSDm", "ETHUSD", "ETHUSDc"]
        elif s in ["US30", "DJ30", "DOW"]:
            candidates = ["US30m", "US30", "DJ30"]
        elif s in ["NAS100", "USTEC", "NDX"]:
            candidates = ["USTECm", "NAS100m", "USTEC", "NAS100"]
        else:
            clean = s.replace("/", "").replace("-", "")
            candidates = [clean, clean + "m", clean + "c", clean.lower()]

        for c in candidates:
            if mt5.symbol_info(c) is not None:
                mt5.symbol_select(c, True)
                return c
        return candidates[0]

    def parse_trading_signal(self, text: str):
        """
        Parses pasted text or command to extract trade parameters:
        action (BUY/SELL), symbol, sl, tp, entry, lot.
        """
        if not text or not isinstance(text, str):
            return None
        raw = text.strip()
        clean = raw
        if clean.lower().startswith(("/signal", "/sig", "/buy", "/sell")):
            cmd = clean.split()[0].lower()
            clean = clean[len(cmd):].strip()
            if cmd == "/buy" and not re.search(r'\b(BUY|LONG)\b', clean, re.I):
                clean = "BUY " + clean
            elif cmd == "/sell" and not re.search(r'\b(SELL|SHORT)\b', clean, re.I):
                clean = "SELL " + clean

        # 1. Action: BUY vs SELL
        action = None
        buy_match = re.search(r'\b(BUY|LONG)\b', clean, re.I)
        sell_match = re.search(r'\b(SELL|SHORT)\b', clean, re.I)
        if buy_match and not sell_match:
            action = "BUY"
        elif sell_match and not buy_match:
            action = "SELL"
        elif buy_match and sell_match:
            action = "BUY" if buy_match.start() < sell_match.start() else "SELL"

        if not action:
            return None

        # 2. Symbol
        sym_match = re.search(r'\b(GOLD|XAUUSDm?|XAU|BTCUSDm?|BTC|BITCOIN|ETHUSDm?|ETH|US30m?|NAS100m?|[A-Z]{6}m?)\b', clean, re.I)
        sym_candidate = sym_match.group(1).upper() if sym_match else "GOLD"
        if sym_candidate in ["BUY", "SELL", "LONG", "SHORT", "STOP", "LOSS", "TAKE", "PROFIT", "ENTRY", "NOW"]:
            sym_candidate = "GOLD"
        symbol = self.match_broker_symbol(sym_candidate)

        # 3. Stop Loss (SL)
        sl_match = re.search(r'(?:SL|STOP\s*LOSS|STOP)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?)', clean, re.I)
        sl = float(sl_match.group(1)) if sl_match else None

        # 4. Take Profit (TP or TP1)
        tp_match = re.search(r'(?:TP\s*1?|TAKE\s*PROFIT\s*1?|TARGET\s*1?|TP)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?)', clean, re.I)
        tp = float(tp_match.group(1)) if tp_match else None

        # 5. Entry
        entry_match = re.search(r'(?:ENTRY|PRICE|AT|@)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?)', clean, re.I)
        entry = float(entry_match.group(1)) if entry_match else None

        # 6. Lot Size
        lot_match = re.search(r'(?:LOT|LOTS|VOL|VOLUME|SIZE)\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?)', clean, re.I)
        lot = float(lot_match.group(1)) if lot_match else None

        has_params = (sl is not None) or (tp is not None) or (entry is not None) or ("NOW" in clean.upper()) or (sym_match and sym_candidate in ["GOLD", "XAU", "XAUUSD", "BTC", "BTCUSD"])
        if not has_params:
            return None

        return {
            "action": action,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
            "entry": entry,
            "lot": lot,
            "raw_text": raw
        }

    def prompt_signal_confirmation(self, message, sig_data: dict):
        """
        Presents an institutional confirmation card with risk/reward calculation
        and an instant 1-tap Telegram inline button to execute the trade.
        """
        bot = self.bot
        cid = message.chat.id
        symbol = sig_data["symbol"]
        action = sig_data["action"]

        info = mt5.symbol_info(symbol)
        if not info:
            bot.reply_to(message, f"❌ Instrument <code>{symbol}</code> is not supported or recognized by your broker.")
            return

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            bot.reply_to(message, f"❌ Market is currently closed or no price feed available for <code>{symbol}</code>.")
            return

        curr_price = tick.ask if action == "BUY" else tick.bid
        acc = mt5.account_info()
        balance = acc.balance if acc else 100.0
        is_small = (balance < 100.0)

        # Lot calculation
        if sig_data.get("lot") is not None:
            lot = max(info.volume_min, round(float(sig_data["lot"]), 2))
        elif is_small:
            lot = info.volume_min
        else:
            if sig_data.get("sl"):
                stop_dist = abs(curr_price - sig_data["sl"])
                risk_pct = self.settings.get("risk_percent", 1.5)
                risk_cash = balance * (risk_pct / 100.0)
                contract_size = info.trade_contract_size if info.trade_contract_size > 0 else 100
                raw_lot = risk_cash / (stop_dist * contract_size) if (stop_dist * contract_size) > 0 else info.volume_min
                lot = max(info.volume_min, min(info.volume_max, round(raw_lot, 2)))
            else:
                lot = info.volume_min

        sig_data["lot"] = lot

        # Distances and risk calculations
        warning = ""
        contract_size = info.trade_contract_size if info.trade_contract_size > 0 else 100

        if sig_data.get("sl"):
            sl_val = sig_data["sl"]
            sl_dist = abs(curr_price - sl_val)
            est_risk = sl_dist * contract_size * lot
            sl_str = f"<code>${sl_val:,.2f}</code> (-{sl_dist:.2f} pts | Est. Risk: -${est_risk:,.2f})"
            if action == "BUY" and sl_val >= curr_price:
                warning += "⚠️ <i>Note: SL is placed above current ask price!</i>\n"
            elif action == "SELL" and sl_val <= curr_price:
                warning += "⚠️ <i>Note: SL is placed below current bid price!</i>\n"
        else:
            sl_str = "<i>None (Uncapped Risk ⚠️)</i>"

        if sig_data.get("tp"):
            tp_val = sig_data["tp"]
            tp_dist = abs(curr_price - tp_val)
            est_gain = tp_dist * contract_size * lot
            tp_str = f"<code>${tp_val:,.2f}</code> (+{tp_dist:.2f} pts | Est. Gain: +${est_gain:,.2f})"
        else:
            tp_str = "<i>None (Open Target Runner)</i>"

        if sig_data.get("sl") and sig_data.get("tp") and sl_dist > 0:
            rr_str = f"1 : {tp_dist / sl_dist:.2f}"
        else:
            rr_str = "Open"

        sig_id = f"{int(time.time() * 1000) % 1000000}"
        self.pending_signals[sig_id] = sig_data

        markup = types.InlineKeyboardMarkup(row_width=2)
        dir_emoji = "📈" if action == "BUY" else "📉"
        markup.add(
            types.InlineKeyboardButton(f"🚀 Execute {action} ({lot:.2f}L)", callback_data=f"sig_exec_{sig_id}"),
            types.InlineKeyboardButton("❌ Cancel", callback_data=f"sig_canc_{sig_id}")
        )

        card = (
            "🎯 <b>TRADE SIGNAL PARSED & READY</b>\n\n"
            f"{warning}"
            f"• <b>Direction:</b> <b>{dir_emoji} {action} {symbol}</b>\n"
            f"• <b>Current Market:</b> <code>${curr_price:,.2f}</code>\n"
            f"• <b>Stop Loss (SL):</b> {sl_str}\n"
            f"• <b>Take Profit (TP):</b> {tp_str}\n"
            f"• <b>Risk / Reward:</b> <b>{rr_str}</b>\n"
            f"• <b>Position Size:</b> <code>{lot:.2f} lot</code> {'<i>(Small-Account Guard Active)</i>' if is_small else ''}\n"
            f"• <b>Account Balance:</b> <code>${balance:,.2f}</code>\n\n"
            "<i>Authorize instant execution by tapping the button below:</i>"
        )
        bot.reply_to(message, card, reply_markup=markup)

    def execute_signal_trade(self, sig_data, user_name="Trader"):
        """
        Sends deal order to MetaTrader 5 within the bot's magic range.
        """
        symbol = sig_data["symbol"]
        action = sig_data["action"]
        sl = float(sig_data.get("sl") or 0.0)
        tp = float(sig_data.get("tp") or 0.0)
        lot = float(sig_data.get("lot") or 0.01)

        info = mt5.symbol_info(symbol)
        if not info:
            return False, f"Symbol '{symbol}' not found on broker.", None

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False, f"Could not retrieve live price for '{symbol}'.", None

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == "BUY" else tick.bid

        # Broker filling mode
        filling_type = mt5.ORDER_FILLING_IOC
        flags = info.filling_mode
        if flags & mt5.ORDER_FILLING_IOC:
            filling_type = mt5.ORDER_FILLING_IOC
        elif flags & mt5.ORDER_FILLING_FOK:
            filling_type = mt5.ORDER_FILLING_FOK
        elif flags & mt5.ORDER_FILLING_RETURN:
            filling_type = mt5.ORDER_FILLING_RETURN

        signal_magic = get_signal_magic(symbol)
        clean_user = re.sub(r'[^a-zA-Z0-9]', '', user_name)[:8]
        comment = f"Sig_{clean_user}"

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 30,
            "magic": signal_magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, res.order, price
        else:
            err = res.comment if res else str(mt5.last_error())
            retcode = res.retcode if res else "Unknown"
            return False, f"{err} (code {retcode})", None

    def _record_token_usage(self, prompt_tokens, candidates_tokens):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.token_usage.get("last_date") != today:
            self.token_usage["requests_today"] = 0
            self.token_usage["last_date"] = today

        self.token_usage["requests_today"] = self.token_usage.get("requests_today", 0) + 1
        self.token_usage["total_prompt_tokens"] = self.token_usage.get("total_prompt_tokens", 0) + prompt_tokens
        self.token_usage["total_candidates_tokens"] = self.token_usage.get("total_candidates_tokens", 0) + candidates_tokens
        self.token_usage["total_tokens"] = self.token_usage.get("total_tokens", 0) + (prompt_tokens + candidates_tokens)

        self.config["token_usage"] = self.token_usage
        save_config(self.config)

    def build_settings_card(self, chat_id=None):
        s = self.settings
        be_active = s.get("breakeven_enabled", True)
        tr_active = s.get("trailing_enabled", True)
        news_active = s.get("news_filter_enabled", True)
        status_str = "PAUSED ⏸️" if self.is_paused else "ACTIVE 🟢"
        curr_risk = float(s.get("risk_percent", 1.5))
        curr_gold = round(float(s.get("gold_threshold", 0.65)) * 100)
        curr_btc = round(float(s.get("btc_threshold", 0.65)) * 100)

        curr_losses = int(s.get("max_daily_losses", 2))
        curr_loss_pct = float(s.get("max_daily_loss_pct", 3.0))

        raw_key = self.config.get("gemini_api_key", "").strip()
        key_preview = f"{raw_key[:7]}...{raw_key[-4:]} (Active ✅)" if len(raw_key) > 10 else "Not Configured ⚠️"

        tu = self.token_usage
        req_today = tu.get("requests_today", 0)
        req_left = max(0, 1500 - req_today)
        pct_left = (req_left / 1500.0) * 100.0

        role_str = ""
        if chat_id is not None:
            r = self.get_user_role(chat_id)
            if r == ROLE_GOD:
                r_badge = "👑 GOD (Full Control)"
            elif r == ROLE_ADMIN:
                r_badge = "🛠️ ADMIN (Full Control)"
            else:
                r_badge = "👁️ VIEWER (Read-Only)"
            role_str = f"• <b>Your Access:</b> <b>{r_badge}</b>\n"

        text = (
            "🎛️ <b>INTERACTIVE BOT CONTROL PANEL</b>\n\n"
            f"{role_str}"
            f"• <b>Bot State:</b> {status_str}\n"
            f"• <b>Risk Per Signal:</b> <b>{curr_risk:.1f}%</b>\n"
            f"• <b>Gold AI Threshold:</b> <b>{curr_gold}%</b>\n"
            f"• <b>BTC AI Threshold:</b> <b>{curr_btc}%</b>\n"
            f"• <b>Max Positions:</b> <b>{s.get('max_positions', 2)}</b>\n"
            f"• <b>Max Daily Losses:</b> <b>{curr_losses} setups</b> ({curr_loss_pct:.1f}% max DD)\n"
            f"• <b>Auto-Breakeven:</b> {'Enabled 🛡️' if be_active else 'Disabled ❌'}\n"
            f"• <b>Trailing Stop:</b> {'Enabled 📈' if tr_active else 'Disabled ❌'}\n"
            f"• <b>News Shield:</b> {'Enabled 📰' if news_active else 'Disabled ❌'}\n"
            f"• <b>Gemini Key:</b> <code>{key_preview}</code>\n"
            f"• <b>Daily Quota Left:</b> <b>{req_left} / 1,500</b> calls ({pct_left:.1f}%)\n\n"
            "<i>💡 Tap any button below to adjust settings instantly:</i>"
        )

        markup = types.InlineKeyboardMarkup()

        # 1. Safety Toggles
        btn_be = types.InlineKeyboardButton(
            f"🛡️ BE: {'ON ✅' if be_active else 'OFF ❌'}",
            callback_data="cfg_toggle_be"
        )
        btn_tr = types.InlineKeyboardButton(
            f"📈 Trail: {'ON ✅' if tr_active else 'OFF ❌'}",
            callback_data="cfg_toggle_tr"
        )
        btn_news = types.InlineKeyboardButton(
            f"📰 News: {'ON ✅' if news_active else 'OFF ❌'}",
            callback_data="cfg_toggle_news"
        )
        markup.row(btn_be, btn_tr, btn_news)

        # 2. Risk Selection Presets
        r_btns = []
        for r in [1.0, 1.5, 2.0, 2.5]:
            mark = "● " if abs(curr_risk - r) < 0.05 else ""
            r_btns.append(types.InlineKeyboardButton(f"{mark}{r}%", callback_data=f"cfg_risk_{r}"))
        markup.row(*r_btns)

        # 3. Gold Threshold Selection
        g_btns = []
        for g in [60, 65, 70]:
            mark = "● " if curr_gold == g else ""
            g_btns.append(types.InlineKeyboardButton(f"🟡 {mark}{g}%", callback_data=f"cfg_gold_{g}"))
        markup.row(*g_btns)

        # 4. BTC Threshold Selection
        b_btns = []
        for b in [60, 65, 70]:
            mark = "● " if curr_btc == b else ""
            b_btns.append(types.InlineKeyboardButton(f"₿ {mark}{b}%", callback_data=f"cfg_btc_{b}"))
        markup.row(*b_btns)

        # 5. Max Daily Losses Presets
        l_btns = []
        for l in [1, 2, 3, 5]:
            mark = "● " if curr_losses == l else ""
            l_btns.append(types.InlineKeyboardButton(f"🛑 {mark}{l} Loss{'es' if l > 1 else ''}", callback_data=f"cfg_losses_{l}"))
        markup.row(*l_btns)

        # 5. Quick Account Switch
        markup.row(
            types.InlineKeyboardButton("🧪 Switch Demo", callback_data="cfg_acc_demo"),
            types.InlineKeyboardButton("💰 Switch Live", callback_data="cfg_acc_live")
        )

        # 6. Safety & Reset buttons
        markup.row(
            types.InlineKeyboardButton("🔓 Reset Daily Loss Guard", callback_data="cfg_reset_guard")
        )

        # 7. Action buttons
        markup.row(
            types.InlineKeyboardButton("🔑 Set AI Key", callback_data="cfg_ask_gemini"),
            types.InlineKeyboardButton("🔄 Restart Bot", callback_data="cfg_restart"),
            types.InlineKeyboardButton("🔄 Refresh", callback_data="cfg_refresh")
        )

        return text, markup

    def build_account_card(self, chat_id=None):
        acc = mt5.account_info()
        if not acc:
            return "❌ Not connected to MT5. Please check MT5 connection.", None

        mode_str = "DEMO 🧪" if acc.trade_mode == 0 else ("REAL / LIVE 💰" if acc.trade_mode == 2 else "CONTEST 🏆")
        saved_profiles = list(self.config.get("accounts", {}).keys())
        prof_str = ", ".join([f"<code>{p}</code>" for p in saved_profiles]) or "None"

        role_str = ""
        if chat_id is not None:
            r = self.get_user_role(chat_id)
            if r == ROLE_GOD:
                r_badge = "👑 GOD (Full Control)"
            elif r == ROLE_ADMIN:
                r_badge = "🛠️ ADMIN (Full Control)"
            else:
                r_badge = "👁️ VIEWER (Read-Only)"
            role_str = f"• <b>Your Access:</b> <b>{r_badge}</b>\n"

        text = (
            f"👤 <b>ACTIVE MT5 ACCOUNT PROFILE</b>\n\n"
            f"{role_str}"
            f"• <b>Account ID:</b> <code>{acc.login}</code>\n"
            f"• <b>Trading Mode:</b> <b>{mode_str}</b>\n"
            f"• <b>Broker Server:</b> <code>{acc.server}</code>\n"
            f"• <b>Account Name:</b> {acc.name}\n"
            f"• <b>Currency:</b> {acc.currency}\n"
            f"• <b>Leverage:</b> 1:{acc.leverage}\n"
            f"• <b>Balance:</b> ${acc.balance:,.2f}\n"
            f"• <b>Equity:</b> ${acc.equity:,.2f}\n\n"
            f"📁 <b>Saved Profiles:</b> {prof_str}\n\n"
            "<i>💡 Tap below to switch accounts, pause, or refresh:</i>"
        )

        markup = types.InlineKeyboardMarkup()
        markup.row(
            types.InlineKeyboardButton("🧪 Switch to Demo", callback_data="acc_switch_demo"),
            types.InlineKeyboardButton("💰 Switch to Live", callback_data="acc_switch_live")
        )
        markup.row(
            types.InlineKeyboardButton("🔌 Pause / Logout", callback_data="acc_logout"),
            types.InlineKeyboardButton("🔄 Refresh", callback_data="acc_refresh")
        )
        return text, markup

    def _register_handlers(self):
        bot = self.bot

        # --- /start & /menu & /help ---
        @bot.message_handler(commands=["start", "menu", "help"])
        def handle_start(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                if len(self.authorized_chat_ids) == 0:
                    self.authorized_chat_ids.add(cid)
                    self.config["authorized_chat_ids"] = list(self.authorized_chat_ids)
                    save_config(self.config)
                    bot.reply_to(message, f"🎉 <b>Welcome!</b> You have been registered as the Primary Admin! (Chat ID: <code>{cid}</code>)")
                else:
                    bot.reply_to(
                        message,
                        f"⛔ <b>Access Denied</b>\n\n"
                        f"Your Chat ID (<code>{cid}</code>) is not on the authorized whitelist.\n\n"
                        f"Ask the Bot Owner (GOD) to invite you using:\n"
                        f"<code>/adduser {cid}</code>"
                    )
                    return

            role = self.get_user_role(cid)
            if role == ROLE_GOD:
                role_badge = "👑 <b>Supreme Master (GOD)</b>"
                role_desc = "You have full, unrestricted master authority over all algorithmic parameters, trades, MT5 accounts, and user permissions."
            elif role == ROLE_ADMIN:
                role_badge = "🛠️ <b>Administrator</b>"
                role_desc = "You have full trading permissions to execute orders, modify risk/thresholds, and switch accounts."
            else:
                role_badge = "👁️ <b>Guest Viewer (Read-Only)</b>"
                role_desc = "You have read-only access to monitor real-time trade alerts, telemetry, financial balance, and chat with the AI."

            welcome_msg = (
                "🏛️ <b>WELCOME TO PHANTOMREX QUANTITATIVE AI</b>\n"
                "<i>Institutional-Grade Algorithmic Trading System for MetaTrader 5</i>\n\n"
                f"• <b>Your Access Level:</b> {role_badge}\n"
                f"<i>{role_desc}</i>\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "⚡ <b>SYSTEM ARCHITECTURE & CAPABILITIES:</b>\n\n"
                "• 🥇 <b>Dual-Asset Automation:</b> Continuous 24/7 scanning and high-speed autonomous execution on <b>Gold (XAUUSDm)</b> and <b>Bitcoin (BTCUSDm)</b>.\n\n"
                "• 🧠 <b>Machine Learning Models:</b> Direction-invariant Random Forest classifiers trained on relative volume flow, ATR volatility squeezes, and rejection wicks with a <b>95%+ verified precision threshold</b>.\n\n"
                "• 🛡️ <b>Trade Lifecycle Protection:</b> Automated 3-stage scaling (TP1/TP2/TP3), <b>Auto-Breakeven ($0.00 Risk Shield)</b>, and dynamic <b>ATR Trailing Stops</b> on trend runners.\n\n"
                "• 💬 <b>Conversational Telemetry:</b> Integrated Google Gemini AI for instant real-time market audits, volatility analysis, and conversational Q&A.\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "🧭 <b>QUICK CONTROLS & COMMANDS:</b>\n"
                "• <b>📊 Status / 💰 Balance:</b> Live market telemetry, equity, & floating PnL\n"
                "• <b>📋 Positions:</b> View active orders with live profit tracking\n"
                "• <b>⚙️ Settings:</b> One-touch touch panel for risk, AI thresholds, & safety toggles\n"
                "• <b>👤 Account:</b> Seamlessly switch between Demo & Live MT5 accounts\n"
                "• <b>🧠 Analyze:</b> On-demand AI evaluation of Gold or Bitcoin setups\n\n"
                "<i>💡 Use the interactive buttons below, or simply ask me any market question in plain English!</i>"
            )
            bot.send_message(cid, welcome_msg, reply_markup=get_main_keyboard())

        # --- /account ---
        @bot.message_handler(commands=["account"])
        def handle_account(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            text, markup = self.build_account_card(chat_id=cid)
            bot.reply_to(message, text, reply_markup=markup)

        # --- /login ---
        @bot.message_handler(commands=["login"])
        def handle_login(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot switch MT5 accounts.\nAsk the Bot Owner (GOD) for permission.")
                return

            parts = message.text.split()
            if len(parts) < 2:
                bot.reply_to(message, "⚠️ Usage:\n• <code>/login demo</code>\n• <code>/login live</code>\n• <code>/login &lt;account_id&gt; &lt;password&gt; &lt;server&gt;</code>")
                return

            target = parts[1].lower()
            accounts = self.config.get("accounts", {})

            # 1. Switch by saved profile name
            if target in accounts:
                acc_cfg = accounts[target]
                login_id = int(acc_cfg.get("login", 0))
                password = acc_cfg.get("password", "")
                server = acc_cfg.get("server", "")

                if not login_id or not server or not password:
                    bot.reply_to(message, f"⚠️ Incomplete credentials for profile '{target}'.\nSave credentials with:\n<code>/saveaccount {target} &lt;login&gt; &lt;password&gt; &lt;server&gt;</code>")
                    return

                bot.send_chat_action(cid, "typing")
                success = mt5.login(login=login_id, password=password, server=server)
                if success:
                    self.config["active_account"] = target
                    save_config(self.config)
                    acc = mt5.account_info()
                    mode_str = "DEMO 🧪" if acc and acc.trade_mode == 0 else "REAL / LIVE 💰"
                    bot.reply_to(
                        message,
                        f"🎉 <b>Switched to {target.upper()} Account!</b>\n\n"
                        f"• Login: <code>{acc.login}</code>\n"
                        f"• Server: <code>{acc.server}</code>\n"
                        f"• Mode: <b>{mode_str}</b>\n"
                        f"• Balance: ${acc.balance:,.2f}"
                    )
                else:
                    err = mt5.last_error()
                    bot.reply_to(message, f"❌ Failed to login to '{target}' (Error: {err})")
                return

            # 2. Custom credentials on the fly: /login <id> <password> <server>
            if len(parts) >= 4 and parts[1].isdigit():
                login_id = int(parts[1])
                password = parts[2]
                server = parts[3]

                # Delete message to hide password from chat history
                try:
                    bot.delete_message(cid, message.message_id)
                except Exception:
                    pass

                bot.send_chat_action(cid, "typing")
                success = mt5.login(login=login_id, password=password, server=server)
                if success:
                    acc = mt5.account_info()
                    mode_str = "DEMO 🧪" if acc and acc.trade_mode == 0 else "REAL / LIVE 💰"
                    bot.send_message(
                        cid,
                        f"🎉 <b>Successfully Connected to MT5!</b>\n\n"
                        f"• Login: <code>{acc.login}</code>\n"
                        f"• Server: <code>{acc.server}</code>\n"
                        f"• Mode: <b>{mode_str}</b>\n"
                        f"• Balance: ${acc.balance:,.2f}\n"
                        f"<i>(Password message was deleted for security)</i>"
                    )
                else:
                    err = mt5.last_error()
                    bot.send_message(cid, f"❌ Login failed: {err}")
                return

            avail = ", ".join(accounts.keys()) or "None"
            bot.reply_to(message, f"⚠️ Profile '{target}' not found. Available profiles: {avail}\nOr use: <code>/login &lt;id&gt; &lt;pass&gt; &lt;server&gt;</code>")

        # --- /saveaccount ---
        @bot.message_handler(commands=["saveaccount"])
        def handle_saveaccount(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot save account credentials.")
                return

            parts = message.text.split()
            if len(parts) < 5 or not parts[2].isdigit():
                bot.reply_to(message, "⚠️ Usage: <code>/saveaccount &lt;profile_name&gt; &lt;login&gt; &lt;password&gt; &lt;server&gt;</code>\nExample: <code>/saveaccount demo 12345678 MyPass Exness-MT5Trial9</code>")
                return

            name = parts[1].lower()
            login_id = int(parts[2])
            password = parts[3]
            server = parts[4]

            try:
                bot.delete_message(cid, message.message_id)
            except Exception:
                pass

            if "accounts" not in self.config:
                self.config["accounts"] = {}

            self.config["accounts"][name] = {
                "login": login_id,
                "password": password,
                "server": server
            }
            save_config(self.config)
            bot.send_message(
                cid,
                f"✅ Profile <b>'{name}'</b> saved successfully!\n"
                f"• Login: <code>{login_id}</code>\n"
                f"• Server: <code>{server}</code>\n\n"
                f"You can now switch to it anytime with: <code>/login {name}</code>\n"
                f"<i>(Password message was deleted for security)</i>"
            )

        # --- /retrain ---
        @bot.message_handler(commands=["retrain"])
        def handle_retrain(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ Read-Only Mode: You cannot trigger model retraining.")
                return

            parts = message.text.split()
            target = parts[1].lower() if len(parts) > 1 else None
            sym_label = "Gold & Bitcoin" if not target else ("Gold" if "gold" in target or "xau" in target else "Bitcoin")

            import train_model
            candles_cnt, _ = train_model.get_history_candle_counts()
            bot.reply_to(
                message,
                f"🧠 <b>Initiating AI Self-Learning Engine for {sym_label}...</b>\n\n"
                f"• Pulling freshest {candles_cnt:,} candles from MetaTrader 5\n"
                "• Simulating every setup (SL, breakeven, TP, spread costs)\n"
                "• Purged walk-forward validation (slow vs fast trend filter)\n"
                "• Picking cutoff on out-of-sample results only\n\n"
                "<i>This runs in background (~3–5 min). You will receive an alert once complete!</i>"
            )

            def _worker():
                try:
                    import train_model
                    res_list = train_model.run_retrain_task(target_symbol=target)
                    lines = [f"🎉 <b>AI RETRAINING COMPLETE! 🧠</b>\n"]
                    promoted_any = False
                    for r in res_list:
                        if r.get("success"):
                            sym = r["symbol"]
                            w_rate = r["win_rate"]
                            th = r["recommended_threshold"] * 100
                            samples = r["total_samples"]
                            if r.get("edge_ok"):
                                edge = "✅ PASS — trading enabled"
                            elif r.get("status") == "INSUFFICIENT_DATA":
                                edge = f"⚠️ INSUFFICIENT_DATA ({r.get('holdout_trades', 0)}/30 holdout trades) — new entries disabled"
                            else:
                                edge = "⛔ FAIL — new entries disabled"

                            if r.get("promoted"):
                                promoted_any = True
                                promo_badge = f"🏆 <b>PROMOTED</b> ({r.get('promotion_reason', '')})"
                            else:
                                promo_badge = f"🛑 <b>REJECTED BY GATE</b> ({r.get('promotion_reason', '')})"

                            lines.append(
                                f"• <b>{sym}</b>: {samples:,} setups simulated\n"
                                f"  - Trend filter: <b>{r.get('trend_mode', 'slow')}</b>\n"
                                f"  - Cutoff: <b>{th:.0f}%</b> | Holdout trades: {r.get('holdout_trades', 0)} (Dev OOS: {r.get('oos_trades', 0)})\n"
                                f"  - Holdout Win Rate: <b>{w_rate:.1f}%</b> | Exp: <b>{r.get('holdout_expectancy_r', 0):+.3f}R</b> (Thinned t: {r.get('holdout_thinned_t', 0):+.2f})\n"
                                f"  - Dev OOS Expectancy: <b>{r.get('oos_expectancy_r', 0):+.3f}R</b> | +Folds: {r.get('positive_folds', 0)}/{r.get('n_folds', 0)}\n"
                                f"  - Edge check: <b>{edge}</b>\n"
                                f"  - Promotion gate: {promo_badge}\n"
                            )
                        else:
                            lines.append(f"• <b>{r.get('symbol')}</b>: ⚠️ {r.get('error')}\n")

                    if promoted_any:
                        lines.append("🏆 <i>Promoted models hot-reloaded into live memory (previous versions saved as .prev.pkl).</i>")
                    else:
                        lines.append("ℹ️ <i>Candidate models did not pass promotion gate. Incumbent models retained untouched.</i>")
                    report_text = "\n".join(lines)
                    self.send_broadcast_alert(report_text)

                    if promoted_any and self.on_retrain_callback:
                        self.on_retrain_callback(target)

                except Exception as e:
                    self.send_broadcast_alert(f"⚠️ Retraining error: {e}")

            threading.Thread(target=_worker, daemon=True).start()

        # --- /rollback ---
        @bot.message_handler(commands=["rollback"])
        def handle_rollback(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ Read-Only Mode: You cannot trigger model rollbacks.")
                return

            import model_promotion

            parts = message.text.split()
            target = parts[1].lower() if len(parts) > 1 else None

            targets_to_check = []
            if not target or "gold" in target or "xau" in target:
                targets_to_check.append(("Gold (XAUUSDm)", "gold_ai_model.pkl", "gold"))
            if not target or "btc" in target or "bitcoin" in target:
                targets_to_check.append(("Bitcoin (BTCUSDm)", "btc_ai_model.pkl", "btc"))

            lines = ["⏪ <b>AI MODEL ROLLBACK RESULTS</b>\n"]
            rolled_back_any = False

            for label, model_file, sym_key in targets_to_check:
                res = model_promotion.rollback_model(model_file)
                if res.get("success"):
                    rolled_back_any = True
                    lines.append(
                        f"• <b>{label}:</b> ✅ Restored from <code>{model_file}.prev.pkl</code>\n"
                        f"  - Trained: <b>{res.get('trained_date')}</b>\n"
                        f"  - Cutoff: <b>{res.get('threshold', 65.0):.0f}%</b> | Trend: <b>{res.get('trend_mode')}</b>\n"
                        f"  - Edge check: <b>{'PASS' if res.get('edge_ok') else 'DISABLED'}</b>\n"
                    )
                else:
                    lines.append(f"• <b>{label}:</b> ⚠️ {res.get('error')}\n")

            if rolled_back_any:
                lines.append("🔄 <i>Hot-reloading rolled-back model(s) into live trading engine...</i>")
                if self.on_retrain_callback:
                    self.on_retrain_callback(target)
            else:
                lines.append("ℹ️ <i>No live models were modified.</i>")

            bot.reply_to(message, "\n".join(lines))

        # --- /logout ---
        @bot.message_handler(commands=["logout"])
        def handle_logout(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot disconnect or pause trading.")
                return
            self.is_paused = True
            self.config["is_paused"] = True
            save_config(self.config)
            bot.reply_to(message, "🔌 <b>MT5 Session Paused / Logged Out.</b>\nTrading is paused. Use <code>/login &lt;profile&gt;</code> or <code>/resume</code> to re-activate.")

        # --- /restart ---
        @bot.message_handler(commands=["restart"])
        def handle_restart(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot restart the bot.")
                return

            bot.reply_to(
                message,
                "🔄 <b>Restarting Trading Bot Engine...</b>\n\n"
                "• Reloading fresh Python source files\n"
                "• Re-connecting to MetaTrader 5\n"
                "• Bot will be back online in ~5 seconds!"
            )
            self.trigger_restart(delay=1.5)

        # --- /resetguard ---
        @bot.message_handler(commands=["resetguard"])
        def handle_reset_guard(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.is_god(cid):
                bot.reply_to(message, "⛔ <b>Unauthorized:</b> Only Supreme Master (GOD) can reset the loss guard.")
                return

            self.settings["daily_guard_reset_time"] = time.time()
            self.config["settings"] = self.settings
            save_config(self.config)
            bot.reply_to(
                message,
                "🔓 <b>Daily Loss Guard Reset!</b>\n\n"
                "• All prior losses for today have been cleared from the circuit breaker.\n"
                "• The bot is now unlocked and scanning for new high-confidence setups!"
            )

        # --- /update ---
        @bot.message_handler(commands=["update"])
        def handle_update(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot update the bot.")
                return

            if not self.is_god(cid):
                bot.reply_to(message, "👁️ <b>Unauthorized.</b>")
                return

            project_dir = os.path.dirname(os.path.abspath(__file__))
            git_dir = os.path.join(project_dir, ".git")

            if not os.path.exists(git_dir):
                bot.reply_to(
                    message,
                    "ℹ️ <b>Git repository not linked on this server (.git folder missing).</b>\n\n"
                    "You can update files directly by sending them as document attachments to this chat!"
                )
                return

            git_bin = shutil.which("git")
            if not git_bin:
                candidates = [
                    r"C:\Program Files\Git\cmd\git.exe",
                    r"C:\Program Files\Git\bin\git.exe",
                    r"C:\Program Files (x86)\Git\cmd\git.exe",
                    r"C:\Program Files (x86)\Git\bin\git.exe",
                    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe"),
                    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Git\cmd\git.exe"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        git_bin = c
                        break

            if not git_bin:
                bot.reply_to(
                    message,
                    "⚠️ <b>Git is not installed on this VPS!</b> (WinError 2)\n\n"
                    "• The <code>git.exe</code> executable was not found on your VPS PATH.\n"
                    "• <b>Option 1:</b> Install Git on your VPS from https://git-scm.com/download/win\n"
                    "• <b>Option 2:</b> Send any updated <code>.py</code> file directly to this chat to update it instantly!"
                )
                return

            bot.reply_to(message, "📥 <b>Checking for updates from GitHub...</b>")
            try:
                stash_res = subprocess.run(
                    [git_bin, "stash", "push", "-m", "preserve local changes before bot update"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if stash_res.returncode != 0:
                    out = (stash_res.stdout + stash_res.stderr).strip()
                    bot.reply_to(
                        message,
                        "⚠️ <b>Update cancelled:</b> Git could not preserve local changes in a stash.\n\n"
                        f"<pre>{out[:400]}</pre>",
                    )
                    return

                status_res = subprocess.run(
                    [git_bin, "status", "--porcelain"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if status_res.returncode != 0 or status_res.stdout.strip():
                    out = (status_res.stdout + status_res.stderr).strip()
                    bot.reply_to(
                        message,
                        "⚠️ <b>Update cancelled:</b> The working tree is not clean after stashing. "
                        "No pull was attempted; inspect and preserve these files first.\n\n"
                        f"<pre>{out[:400]}</pre>",
                    )
                    return

                res = subprocess.run([git_bin, "pull"], cwd=project_dir, capture_output=True, text=True, timeout=30)
                out = (res.stdout + res.stderr).strip()
                if "Already up to date" in out:
                    bot.reply_to(
                        message,
                        "✅ <b>System is already up to date.</b>\n"
                        "Any local tracked changes remain preserved in Git stash; do not pop them over changed files.",
                    )
                elif res.returncode == 0:
                    bot.reply_to(
                        message,
                        f"🎉 <b>Update downloaded!</b>\n\n<pre>{out[:300]}</pre>\n\n"
                        "Any local tracked changes remain preserved in Git stash. "
                        "Do not pop them over changed files; merge needed data manually.\n\n"
                        "🔄 Restarting engine now...",
                    )
                    self.trigger_restart(delay=2.0)
                else:
                    bot.reply_to(
                        message,
                        f"⚠️ <b>Git pull reported an issue:</b>\n\n<pre>{out[:400]}</pre>\n\n"
                        "Local tracked changes remain preserved in Git stash.",
                    )
            except Exception as ex:
                bot.reply_to(message, f"❌ Update error: {ex}")

        # --- /news ---
        @bot.message_handler(commands=["news"])
        def handle_news(message):
            self.send_news_summary(message)

        # --- Direct Document Upload (Restricted to GOD user only, zero internal info leaked) ---
        @bot.message_handler(content_types=["document"])
        def handle_document_upload(message):
            cid = message.chat.id
            # Strict security: Only the owner (GOD) can upload files. Ignore everyone else silently.
            if not self.is_god(cid):
                return

            doc = message.document
            fname = doc.file_name

            allowed_files = [
                "bot.py", "train_model.py", "ai_features.py",
                "telegram_manager.py", "news_filter.py", "telegram_config.json",
                "model_promotion.py", "db_ledger.py"
            ]

            if fname not in allowed_files:
                bot.reply_to(message, "⚠️ Unsupported update package.")
                return

            try:
                bot.reply_to(message, "⏳ <b>Applying update...</b>")
                file_info = bot.get_file(doc.file_id)
                downloaded_file = bot.download_file(file_info.file_path)

                project_dir = os.path.dirname(os.path.abspath(__file__))
                target_path = os.path.join(project_dir, fname)

                # Save backup of current version
                if os.path.exists(target_path):
                    shutil.copyfile(target_path, target_path + ".bak")

                with open(target_path, "wb") as f:
                    f.write(downloaded_file)

                bot.send_message(cid, "✅ <b>Update applied successfully!</b>\n\n🔄 Restarting engine now...")
                self.trigger_restart(delay=2.0)

            except Exception as ex:
                bot.reply_to(message, f"❌ Update failed: {ex}")

        # --- /protect ---
        @bot.message_handler(commands=["protect"])
        def handle_protect(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ Read-Only Mode: You cannot modify positions.")
                return

            parts = message.text.split()
            # If user specifies: /protect <ticket> <sl> <tp>
            if len(parts) >= 4 and parts[1].isdigit():
                ticket = int(parts[1])
                try:
                    sl_val = float(parts[2])
                    tp_val = float(parts[3])
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": sl_val,
                        "tp": tp_val,
                    }
                    res = mt5.order_send(req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        bot.reply_to(message, f"✅ <b>Applied SL/TP to #{ticket} with your authority!</b>\n• SL: <code>${sl_val:.2f}</code>\n• TP: <code>${tp_val:.2f}</code>")
                    else:
                        bot.reply_to(message, f"❌ Failed to set SL/TP: {res.comment if res else mt5.last_error()}")
                except ValueError:
                    bot.reply_to(message, "⚠️ Invalid numbers. Usage: <code>/protect &lt;ticket&gt; &lt;sl&gt; &lt;tp&gt;</code>")
                return

            # If user specifies: /protect <ticket>
            if len(parts) == 2 and parts[1].isdigit():
                ticket = int(parts[1])
                self.check_and_prompt_unprotected_trades(specific_ticket=ticket, target_chat_id=cid)
                return

            # Default: /protect (scan all open trades)
            self.check_and_prompt_unprotected_trades(target_chat_id=cid)

        # --- /signal, /buy, /sell ---
        @bot.message_handler(commands=["signal", "sig"])
        def handle_signal_command(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            parts = message.text.split(maxsplit=1)
            if len(parts) < 2 or parts[1].strip().lower() in ["help", "guide"]:
                bot.reply_to(
                    message,
                    "🎯 <b>SIGNAL COPILOT GUIDE</b>\n\n"
                    "Paste any trade signal from your channels or type it out:\n"
                    "• <code>BUY GOLD SL 4120 TP 4145</code>\n"
                    "• <code>SELL BTC SL 83500 TP 80000</code>\n"
                    "• <code>BUY XAUUSD @ 4134 SL 4124 TP 4148 LOT 0.01</code>\n\n"
                    "Or multi-line channel format:\n"
                    "<code>GOLD BUY NOW\n"
                    "SL: 4120\n"
                    "TP: 4145</code>\n\n"
                    "⚡ <i>The bot parses price levels, checks risk, and waits for your 1-tap confirmation!</i>"
                )
                return

            sig_data = self.parse_trading_signal(parts[1])
            if sig_data:
                self.prompt_signal_confirmation(message, sig_data)
            else:
                bot.reply_to(message, "⚠️ Could not parse trade parameters. Make sure to specify action (BUY/SELL) and SL/TP levels.\nExample: <code>BUY GOLD SL 4120 TP 4145</code>")

        @bot.message_handler(commands=["buy"])
        def handle_buy_command(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            sig_data = self.parse_trading_signal(message.text)
            if sig_data:
                self.prompt_signal_confirmation(message, sig_data)
            else:
                bot.reply_to(message, "⚠️ Usage: <code>/buy gold sl 4120 tp 4145 [lot 0.01]</code>")

        @bot.message_handler(commands=["sell"])
        def handle_sell_command(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            sig_data = self.parse_trading_signal(message.text)
            if sig_data:
                self.prompt_signal_confirmation(message, sig_data)
            else:
                bot.reply_to(message, "⚠️ Usage: <code>/sell btc sl 83500 tp 80000 [lot 0.01]</code>")

        # --- /settings ---
        @bot.message_handler(commands=["settings"])
        def handle_settings(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            text, markup = self.build_settings_card(chat_id=cid)
            bot.reply_to(message, text, reply_markup=markup)

        # --- Inline Interactive Buttons Callback Handler ---
        @bot.callback_query_handler(func=lambda call: True)
        def handle_inline_callbacks(call):
            cid = call.message.chat.id
            mid = call.message.message_id
            if not self.is_authorized(cid):
                bot.answer_callback_query(call.id, "⛔ Access Denied", show_alert=True)
                return

            data = call.data

            # Role-Based Permission Check: Read-Only Viewers cannot modify settings or switch accounts
            if not self.can_control(cid):
                if data in ["acc_refresh", "cfg_refresh", "cfg_tokens"]:
                    pass
                else:
                    bot.answer_callback_query(
                        call.id,
                        "👁️ Read-Only Mode: You cannot modify bot settings or switch accounts.",
                        show_alert=True
                    )
                    return

            # 1. Breakeven Toggle
            if data == "cfg_toggle_be":
                val = not self.settings.get("breakeven_enabled", True)
                self.settings["breakeven_enabled"] = val
                self.config["settings"] = self.settings
                save_config(self.config)
                bot.answer_callback_query(call.id, f"🛡️ Breakeven is now {'ON ✅' if val else 'OFF ❌'}")
                text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 2. Trailing Stop Toggle
            elif data == "cfg_toggle_tr":
                val = not self.settings.get("trailing_enabled", True)
                self.settings["trailing_enabled"] = val
                self.config["settings"] = self.settings
                save_config(self.config)
                bot.answer_callback_query(call.id, f"📈 Trailing Stop is now {'ON ✅' if val else 'OFF ❌'}")
                text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # News Shield Toggle
            elif data == "cfg_toggle_news":
                val = not self.settings.get("news_filter_enabled", True)
                self.settings["news_filter_enabled"] = val
                self.config["settings"] = self.settings
                save_config(self.config)
                bot.answer_callback_query(call.id, f"📰 News Shield is now {'ON ✅' if val else 'OFF ❌'}")
                text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 3. Risk Presets
            elif data.startswith("cfg_risk_"):
                try:
                    val = float(data.split("_")[2])
                    self.settings["risk_percent"] = val
                    self.config["settings"] = self.settings
                    save_config(self.config)
                    bot.answer_callback_query(call.id, f"🎯 Risk set to {val:.1f}%")
                    text, markup = self.build_settings_card(chat_id=cid)
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 4. Gold AI Threshold Presets
            elif data.startswith("cfg_gold_"):
                try:
                    val = int(data.split("_")[2])
                    self.settings["gold_threshold"] = val / 100.0
                    self.config["settings"] = self.settings
                    save_config(self.config)
                    bot.answer_callback_query(call.id, f"🟡 Gold Threshold set to {val}%")
                    text, markup = self.build_settings_card(chat_id=cid)
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 5. BTC AI Threshold Presets
            elif data.startswith("cfg_btc_"):
                try:
                    val = int(data.split("_")[2])
                    self.settings["btc_threshold"] = val / 100.0
                    self.config["settings"] = self.settings
                    save_config(self.config)
                    bot.answer_callback_query(call.id, f"₿ BTC Threshold set to {val}%")
                    text, markup = self.build_settings_card(chat_id=cid)
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 6. Max Daily Losses Presets
            elif data.startswith("cfg_losses_"):
                try:
                    val = int(data.split("_")[2])
                    self.settings["max_daily_losses"] = val
                    self.config["settings"] = self.settings
                    save_config(self.config)
                    bot.answer_callback_query(call.id, f"🛑 Max Daily Losses set to {val}")
                    text, markup = self.build_settings_card(chat_id=cid)
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 7. Reset Daily Loss Guard (Owner GOD only)
            elif data == "cfg_reset_guard":
                if not self.is_god(cid):
                    bot.answer_callback_query(call.id, "⛔ Only Supreme Master (GOD) can reset the loss guard.", show_alert=True)
                    return
                self.settings["daily_guard_reset_time"] = time.time()
                self.config["settings"] = self.settings
                save_config(self.config)
                bot.answer_callback_query(call.id, "🔓 Daily Loss Guard reset! Prior losses cleared for today.")
                text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 6. Switch to Demo
            elif data in ["cfg_acc_demo", "acc_switch_demo"]:
                accounts = self.config.get("accounts", {})
                demo_cfg = accounts.get("demo", {})
                login_id = int(demo_cfg.get("login", 0))
                password = demo_cfg.get("password", "")
                server = demo_cfg.get("server", "")

                if not login_id or not server or not password:
                    bot.answer_callback_query(
                        call.id,
                        "⚠️ Demo credentials incomplete! Use /saveaccount demo <login> <pass> <server>",
                        show_alert=True
                    )
                    return

                bot.answer_callback_query(call.id, "🔄 Switching to Demo account...")
                success = mt5.login(login=login_id, password=password, server=server)
                if success:
                    self.config["active_account"] = "demo"
                    save_config(self.config)
                    acc = mt5.account_info()
                    bot.send_message(
                        cid,
                        f"🎉 <b>Switched to DEMO Account! 🧪</b>\n\n"
                        f"• Login: <code>{acc.login}</code>\n"
                        f"• Server: <code>{acc.server}</code>\n"
                        f"• Balance: ${acc.balance:,.2f}"
                    )
                else:
                    err = mt5.last_error()
                    bot.send_message(cid, f"❌ Demo Login failed: {err}")

                if data == "acc_switch_demo":
                    text, markup = self.build_account_card(chat_id=cid)
                else:
                    text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 7. Switch to Live
            elif data in ["cfg_acc_live", "acc_switch_live"]:
                accounts = self.config.get("accounts", {})
                live_cfg = accounts.get("live", {})
                login_id = int(live_cfg.get("login", 0))
                password = live_cfg.get("password", "")
                server = live_cfg.get("server", "")

                if not login_id or not server or not password:
                    bot.answer_callback_query(
                        call.id,
                        "⚠️ Live credentials incomplete! Use /saveaccount live <login> <pass> <server>",
                        show_alert=True
                    )
                    return

                bot.answer_callback_query(call.id, "🔄 Switching to LIVE account...")
                success = mt5.login(login=login_id, password=password, server=server)
                if success:
                    self.config["active_account"] = "live"
                    save_config(self.config)
                    acc = mt5.account_info()
                    bot.send_message(
                        cid,
                        f"🎉 <b>Switched to LIVE Account! 💰</b>\n\n"
                        f"• Login: <code>{acc.login}</code>\n"
                        f"• Server: <code>{acc.server}</code>\n"
                        f"• Balance: ${acc.balance:,.2f}"
                    )
                else:
                    err = mt5.last_error()
                    bot.send_message(cid, f"❌ Live Login failed: {err}")

                if data == "acc_switch_live":
                    text, markup = self.build_account_card(chat_id=cid)
                else:
                    text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 8. Pause / Logout
            elif data == "acc_logout":
                self.is_paused = True
                self.config["is_paused"] = True
                save_config(self.config)
                bot.answer_callback_query(call.id, "🔌 Trading paused / Logged out")
                bot.send_message(cid, "🔌 <b>Trading Paused / MT5 Session Logged Out</b>\nUse /resume or switch accounts to reconnect.")
                text, markup = self.build_account_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 9. Account Refresh
            elif data == "acc_refresh":
                bot.answer_callback_query(call.id, "🔄 Refreshed!")
                text, markup = self.build_account_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 10. Token Popup
            elif data == "cfg_tokens":
                tu = self.token_usage
                today_req = tu.get("requests_today", 0)
                tot_tok = tu.get("total_tokens", 0)
                req_left = max(0, 1500 - today_req)
                bot.answer_callback_query(
                    call.id,
                    f"📊 Tokens: {tot_tok:,} total | {today_req}/1500 calls today ({req_left} left)",
                    show_alert=True
                )

            # 11. Interactive Key Update Prompt
            elif data == "cfg_ask_gemini":
                self.awaiting_input[cid] = "gemini_key"
                bot.answer_callback_query(call.id, "🔑 Send your new Gemini key in chat")
                bot.send_message(
                    cid,
                    "🔑 <b>Paste your new Google Gemini API Key:</b>\n\n"
                    "Simply send your key as the next message in this chat.\n"
                    "The bot will automatically test it, save it, and delete your message to protect your credentials.\n\n"
                    "<i>(Send <code>/cancel</code> anytime to abort)</i>"
                )

            # 12. Engine Restart & Refresh
            elif data == "cfg_restart":
                if not self.is_god(cid):
                    bot.answer_callback_query(call.id, "Unauthorized.")
                    return
                bot.answer_callback_query(call.id, "🔄 Restarting bot engine...")
                handle_restart(call.message)

            elif data == "cfg_refresh":
                bot.answer_callback_query(call.id, "🔄 Refreshed!")
                text, markup = self.build_settings_card(chat_id=cid)
                try:
                    bot.edit_message_text(text, cid, mid, reply_markup=markup, parse_mode="HTML")
                except Exception:
                    pass

            # 13. Trade Protection Authorization (User Authority Confirmation)
            elif data.startswith("prot_ok_"):
                parts = data.split("_")
                ticket = int(parts[2])
                new_sl = float(parts[3])
                new_tp = float(parts[4])

                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": new_tp,
                }
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    bot.answer_callback_query(call.id, f"🛡️ Protection applied to #{ticket}!")
                    try:
                        bot.edit_message_text(
                            f"✅ <b>TRADE PROTECTED WITH YOUR AUTHORITY!</b>\n\n"
                            f"• <b>Ticket:</b> <code>#{ticket}</code>\n"
                            f"• <b>Applied SL:</b> <code>${new_sl:.2f}</code>\n"
                            f"• <b>Applied TP:</b> <code>${new_tp:.2f}</code>\n\n"
                            f"🛡️ <i>Risk is officially capped. Well done!</i>",
                            cid, mid, parse_mode="HTML"
                        )
                    except Exception:
                        pass
                else:
                    err_msg = res.comment if res else str(mt5.last_error())
                    bot.answer_callback_query(call.id, f"❌ Failed: {err_msg}", show_alert=True)

            elif data.startswith("prot_no_"):
                ticket = int(data.split("_")[2])
                bot.answer_callback_query(call.id, f"Understood. #{ticket} left unchanged.")
                try:
                    bot.edit_message_text(
                        f"ℹ️ <b>Manual Trade #{ticket}</b> left unchanged as instructed.",
                        cid, mid, parse_mode="HTML"
                    )
                except Exception:
                    pass

            # 14. Signal Execution Callback
            elif data.startswith("sig_exec_"):
                sig_id = data.split("_")[2]
                sig_data = self.pending_signals.get(sig_id)
                if not sig_data:
                    bot.answer_callback_query(call.id, "⚠️ Signal has expired or already executed.", show_alert=True)
                    return

                if not self.can_control(cid):
                    bot.answer_callback_query(call.id, "👁️ Read-Only Mode: You cannot execute trades.", show_alert=True)
                    return

                user_name = call.from_user.first_name or "Trader"
                ok, res_data, open_price = self.execute_signal_trade(sig_data, user_name=user_name)
                if ok:
                    del self.pending_signals[sig_id]
                    ticket = res_data
                    signal_magic = get_signal_magic(sig_data.get("symbol", ""))
                    bot.answer_callback_query(call.id, f"🚀 Executed #{ticket}!")
                    action_sym = "📈 BUY" if sig_data["action"] == "BUY" else "📉 SELL"
                    sl_fmt = f"${sig_data['sl']:.2f}" if sig_data.get('sl') else "None"
                    tp_fmt = f"${sig_data['tp']:.2f}" if sig_data.get('tp') else "None"
                    try:
                        bot.edit_message_text(
                            f"🚀 <b>SIGNAL TRADE EXECUTED!</b>\n\n"
                            f"• <b>Ticket:</b> <code>#{ticket}</code>\n"
                            f"• <b>Direction:</b> <b>{action_sym} {sig_data['symbol']}</b>\n"
                            f"• <b>Volume:</b> <code>{sig_data['lot']:.2f} lot</code>\n"
                            f"• <b>Fill Price:</b> <code>${open_price:,.2f}</code>\n"
                            f"• <b>Stop Loss:</b> <code>{sl_fmt}</code>\n"
                            f"• <b>Take Profit:</b> <code>{tp_fmt}</code>\n"
                            f"• <b>Authorized By:</b> {user_name}\n\n"
                            f"🛡️ <i>Secured under Magic #{signal_magic} (Signal Copilot).</i>",
                            cid, mid, parse_mode="HTML"
                        )
                    except Exception:
                        pass
                else:
                    bot.answer_callback_query(call.id, f"❌ Execution Failed: {res_data}", show_alert=True)

            elif data.startswith("sig_canc_"):
                sig_id = data.split("_")[2]
                if sig_id in self.pending_signals:
                    del self.pending_signals[sig_id]
                bot.answer_callback_query(call.id, "Signal execution cancelled.")
                try:
                    bot.edit_message_text("❌ <b>Signal trade cancelled.</b> No order was placed.", cid, mid, parse_mode="HTML")
                except Exception:
                    pass

        # --- /tokens & /usage ---
        @bot.message_handler(commands=["tokens", "usage"])
        def handle_tokens(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            tu = self.token_usage
            today = datetime.now().strftime("%Y-%m-%d")
            if tu.get("last_date") != today:
                tu["requests_today"] = 0
                tu["last_date"] = today
                self.config["token_usage"] = tu
                save_config(self.config)

            req_today = tu.get("requests_today", 0)
            req_left = max(0, 1500 - req_today)
            pct_left = (req_left / 1500.0) * 100.0

            total_tok = tu.get("total_tokens", 0)
            prompt_tok = tu.get("total_prompt_tokens", 0)
            cand_tok = tu.get("total_candidates_tokens", 0)

            raw_key = self.config.get("gemini_api_key", "").strip()
            key_status = f"<code>{raw_key[:7]}...{raw_key[-4:]}</code> (Valid ✅)" if len(raw_key) > 10 else "None configured"

            text = (
                "📊 <b>GEMINI AI TOKEN & QUOTA USAGE</b>\n\n"
                f"• <b>Active Key:</b> {key_status}\n"
                f"• <b>Today's Requests:</b> <b>{req_today} / 1,500</b>\n"
                f"• <b>Daily Quota Remaining:</b> <b>{req_left} requests ({pct_left:.1f}% left)</b>\n"
                f"• <b>Total Tokens Consumed:</b> <b>{total_tok:,}</b>\n"
                f"  ├ Input (Prompt) Tokens: {prompt_tok:,}\n"
                f"  └ Output (Response) Tokens: {cand_tok:,}\n\n"
                "💡 <b>Google Gemini Free Tier Limits:</b>\n"
                "• <b>RPD:</b> 1,500 free requests per day\n"
                "• <b>RPM:</b> 15 requests per minute\n"
                "• <b>TPM:</b> 1,000,000 tokens per minute\n\n"
                "<i>To change your key anytime, use:</i> <code>/setgemini &lt;new_key&gt;</code>"
            )
            bot.reply_to(message, text)

        # --- /setgemini (Change API Key) ---
        @bot.message_handler(commands=["setgemini", "setkey"])
        def handle_setgemini(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot change the Gemini API key.\nAsk the Bot Creator (GOD) for permission.")
                return

            parts = message.text.split(maxsplit=1)
            if len(parts) < 2:
                bot.reply_to(message, "⚠️ Usage: <code>/setgemini &lt;your_new_api_key&gt;</code>\nExample: <code>/setgemini AIzaSy...</code>")
                return

            new_key = parts[1].strip()
            bot.send_chat_action(cid, "typing")

            # Validate key with Gemini endpoint
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={new_key}"
            try:
                r = requests.post(test_url, json={"contents": [{"parts": [{"text": "ping"}]}]}, timeout=10)
                if r.status_code == 200:
                    self.config["gemini_api_key"] = new_key
                    save_config(self.config)
                    bot.reply_to(
                        message,
                        f"✅ <b>Gemini API Key Verified & Updated!</b>\n\n"
                        f"• Key: <code>{new_key[:8]}...{new_key[-4:]}</code>\n"
                        f"• Model: <code>gemini-3.1-flash-lite</code>\n"
                        f"• Status: Active & Ready for real-time chat! 🎉"
                    )
                else:
                    bot.reply_to(message, f"❌ <b>Key Validation Failed (HTTP {r.status_code}):</b>\n{r.text[:200]}")
            except Exception as e:
                bot.reply_to(message, f"⚠️ Error validating key: {e}")

        # --- /setrisk ---
        @bot.message_handler(commands=["setrisk"])
        def handle_setrisk(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot change risk parameters.\nAsk the Bot Creator (GOD) for permission.")
                return

            parts = message.text.split()
            if len(parts) < 2:
                bot.reply_to(message, "⚠️ Usage: <code>/setrisk &lt;percent&gt;</code>\nExample: <code>/setrisk 1.5</code>")
                return

            try:
                val = float(parts[1].replace("%", ""))
                if val <= 0 or val > 10.0:
                    bot.reply_to(message, "⚠️ Risk percent must be between 0.1% and 10.0% for safety.")
                    return

                self.settings["risk_percent"] = val
                self.config["settings"] = self.settings
                save_config(self.config)
                bot.reply_to(message, f"✅ <b>Risk Per Signal updated to:</b> <b>{val:.1f}%</b>")
            except ValueError:
                bot.reply_to(message, "⚠️ Invalid number format. Use e.g. <code>/setrisk 1.5</code>")

        # --- /setthreshold ---
        @bot.message_handler(commands=["setthreshold"])
        def handle_setthreshold(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot change AI thresholds.\nAsk the Bot Creator (GOD) for permission.")
                return

            parts = message.text.split()
            if len(parts) < 3:
                bot.reply_to(message, "⚠️ Usage: <code>/setthreshold &lt;gold|btc&gt; &lt;percent&gt;</code>\nExample: <code>/setthreshold gold 62</code>")
                return

            target = parts[1].lower()
            try:
                pct = float(parts[2].replace("%", ""))
                if pct < 50 or pct > 95:
                    bot.reply_to(message, "⚠️ Threshold percent should be between 50% and 95%.")
                    return

                dec_val = round(pct / 100.0, 2)
                if "gold" in target or "xau" in target:
                    self.settings["gold_threshold"] = dec_val
                    label = "Gold (XAUUSDm)"
                elif "btc" in target:
                    self.settings["btc_threshold"] = dec_val
                    label = "Bitcoin (BTCUSDm)"
                else:
                    bot.reply_to(message, "⚠️ Target must be 'gold' or 'btc'.")
                    return

                self.config["settings"] = self.settings
                save_config(self.config)
                bot.reply_to(message, f"✅ <b>{label} AI Threshold updated to:</b> <b>{pct:.0f}%</b>")
            except ValueError:
                bot.reply_to(message, "⚠️ Invalid number format. Use e.g. <code>/setthreshold gold 65</code>")

        # --- /setmaxpos ---
        @bot.message_handler(commands=["setmaxpos"])
        def handle_setmaxpos(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot change max positions.")
                return

            parts = message.text.split()
            if len(parts) < 2 or not parts[1].isdigit():
                bot.reply_to(message, "⚠️ Usage: <code>/setmaxpos &lt;number&gt;</code>\nExample: <code>/setmaxpos 6</code>")
                return

            val = int(parts[1])
            if val < 1 or val > 3:
                bot.reply_to(message, "⚠️ Max positions must be between 1 and 3.")
                return

            self.settings["max_positions"] = val
            self.config["settings"] = self.settings
            save_config(self.config)
            bot.reply_to(message, f"✅ <b>Max positions per asset updated to:</b> <b>{val}</b>")

        # --- /togglebreakeven ---
        @bot.message_handler(commands=["togglebreakeven"])
        def handle_togglebreakeven(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot toggle trade safety features.")
                return

            curr = self.settings.get("breakeven_enabled", True)
            self.settings["breakeven_enabled"] = not curr
            self.config["settings"] = self.settings
            save_config(self.config)
            status = "ENABLED 🛡️" if not curr else "DISABLED ❌"
            bot.reply_to(message, f"✅ <b>Auto-Breakeven is now:</b> {status}")

        # --- /toggletrailing ---
        @bot.message_handler(commands=["toggletrailing"])
        def handle_toggletrailing(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot toggle trade safety features.")
                return

            curr = self.settings.get("trailing_enabled", True)
            self.settings["trailing_enabled"] = not curr
            self.config["settings"] = self.settings
            save_config(self.config)
            status = "ENABLED 📈" if not curr else "DISABLED ❌"
            bot.reply_to(message, f"✅ <b>Dynamic Trailing Stop is now:</b> {status}")

        # --- /status ---
        @bot.message_handler(commands=["status"])
        def handle_status(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            status_lines = ["📊 <b>REAL-TIME MARKET TELEMETRY:</b>\n"]
            pause_str = "⏸️ <b>PAUSED</b>" if self.is_paused else "🟢 <b>ACTIVE</b>"
            status_lines.append(f"<b>Bot Execution:</b> {pause_str}")
            status_lines.append(f"<b>Risk Setting:</b> {self.settings.get('risk_percent', 1.5):.1f}%\n")

            for sym in ["XAUUSDm", "BTCUSDm"]:
                tick = mt5.symbol_info_tick(sym)
                positions = mt5.positions_get(symbol=sym) or []
                thresh = self.settings.get("gold_threshold" if "XAU" in sym else "btc_threshold", 0.65) * 100
                if tick:
                    status_lines.append(f"• <b>{sym}</b>: ${tick.bid:,.2f} | Open: {len(positions)} | Thresh: {thresh:.0f}%")

            bot.reply_to(message, "\n".join(status_lines))

        # --- /balance ---
        @bot.message_handler(commands=["balance"])
        def handle_balance(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            acc = mt5.account_info()
            if not acc:
                bot.reply_to(message, "❌ Could not retrieve account info from MT5.")
                return

            floating = acc.equity - acc.balance
            float_str = f"+${floating:,.2f} 🟢" if floating >= 0 else f"-${abs(floating):,.2f} 🔴"

            text = (
                f"💰 <b>ACCOUNT FINANCIAL STATUS</b>\n\n"
                f"• <b>Balance:</b> ${acc.balance:,.2f}\n"
                f"• <b>Equity:</b> ${acc.equity:,.2f}\n"
                f"• <b>Floating PnL:</b> {float_str}\n"
                f"• <b>Margin Used:</b> ${acc.margin:,.2f}\n"
                f"• <b>Free Margin:</b> ${acc.margin_free:,.2f}\n"
                f"• <b>Margin Level:</b> {acc.margin_level:.1f}%\n"
            )
            bot.reply_to(message, text)

        # --- /positions ---
        @bot.message_handler(commands=["positions"])
        def handle_positions(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            positions = mt5.positions_get()
            if not positions:
                bot.reply_to(message, "ℹ️ No open positions currently active.")
                return

            lines = [f"📋 <b>ACTIVE POSITIONS ({len(positions)}):</b>\n"]
            for pos in positions:
                p_type = "BUY 📈" if pos.type == mt5.ORDER_TYPE_BUY else "SELL 📉"
                pnl = pos.profit
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                lines.append(
                    f"• <b>#{pos.ticket} {pos.symbol}</b> ({p_type})\n"
                    f"  Lot: {pos.volume} | Entry: ${pos.price_open:.2f}\n"
                    f"  Current: ${pos.price_current:.2f} | PnL: <b>{pnl_str}</b>\n"
                    f"  SL: ${pos.sl:.2f} | TP: ${pos.tp:.2f}\n"
                )
            bot.reply_to(message, "\n".join(lines))

        # --- /stats ---
        @bot.message_handler(commands=["stats", "performance"])
        def handle_stats(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            try:
                import db_ledger

                earliest_ts = db_ledger.get_earliest_signal_timestamp()
                if earliest_ts:
                    try:
                        from_date = datetime.fromisoformat(earliest_ts)
                    except Exception:
                        from_date = datetime.now() - timedelta(days=90)
                else:
                    from_date = datetime.now() - timedelta(days=90)

                to_date = datetime.now() + timedelta(days=1)
                deals = mt5.history_deals_get(from_date, to_date) or []

                stats = db_ledger.compute_performance_stats(deals)

                tot = stats["total_closed"]
                matched = stats["matched_count"]
                unmatched = stats["unmatched_count"]
                wr = stats["win_rate_pct"]
                exp_usd = stats["expectancy_usd"]
                exp_r = stats["expectancy_r"]
                avg_win_usd = stats["avg_win_usd"]
                avg_loss_usd = stats["avg_loss_usd"]
                avg_win_r = stats["avg_win_r"]
                avg_loss_r = stats["avg_loss_r"]
                dd_usd = stats["max_drawdown_usd"]
                dd_r = stats["max_drawdown_r"]
                exits = stats["exit_reasons"]

                lines = [
                    "📊 <b>HISTORICAL PERFORMANCE & STATS TELEMETRY</b>",
                    "<i>Full deal-to-signal join on position_id</i>\n",
                    f"• <b>Closed Positions:</b> <b>{tot}</b> (Matched: {matched} | Unmatched: {unmatched})",
                    f"• <b>Win Rate:</b> <b>{wr:.1f}%</b> (Matched closed trades)",
                    f"• <b>Avg Win:</b> +${avg_win_usd:.2f} (+{avg_win_r:.2f}R)",
                    f"• <b>Avg Loss:</b> -${abs(avg_loss_usd):.2f} ({avg_loss_r:.2f}R)",
                    f"• <b>Expectancy:</b> <b>{exp_usd:+.2f}$</b> | <b>{exp_r:+.2f}R</b> per trade",
                    f"• <b>Max Drawdown:</b> -${dd_usd:.2f} (-{dd_r:.2f}R)\n",
                    "🎯 <b>EXIT REASONS:</b>",
                    f"• TP Hits: <b>{exits['pct_tp']:.1f}%</b>",
                    f"• Breakeven (BE): <b>{exits['pct_be']:.1f}%</b>",
                    f"• Stop Loss (SL): <b>{exits['pct_sl']:.1f}%</b>",
                    f"• Other / Manual: <b>{exits['pct_other']:.1f}%</b>",
                    f"• Unmatched: <b>{exits['pct_unmatched']:.1f}%</b>\n",
                    "🧠 <b>CONFIDENCE CALIBRATION TABLE:</b>",
                ]

                for b in stats["calibration"]:
                    b_name = b["bucket"]
                    b_disp = b["display"]
                    lines.append(f"• <b>{b_name}:</b> {b_disp}")

                report_text = "\n".join(lines)
                bot.reply_to(message, report_text)

            except Exception as e:
                bot.reply_to(message, f"⚠️ Error computing stats telemetry: {e}")

        # --- /spstats ---
        @bot.message_handler(commands=["spstats"])
        def handle_small_profit_stats(message):
            if not self.is_authorized(message.chat.id):
                return
            try:
                import small_profit_ledger

                summaries = small_profit_ledger.get_summary()
                direction_summaries = small_profit_ledger.get_direction_summary()
                recent = small_profit_ledger.get_recent_closed(limit=5)
                lines = [
                    "🧪 <b>SMALL-PROFIT DEMO TEST HISTORY</b>",
                    "<i>Net results include realized MT5 profit, commission, swap, and fees.</i>",
                ]
                if not summaries:
                    lines.append("No candidate bars recorded yet.")
                for item in summaries:
                    closed = int(item["closed_trades"] or 0)
                    wins = int(item["wins"] or 0)
                    win_rate = wins / closed * 100 if closed else 0.0
                    expectancy = item["expectancy_usd"]
                    expectancy_text = f"${expectancy:+.3f}" if expectancy is not None else "n/a"
                    lines.append(
                        f"\n<b>{item['strategy']}</b>\n"
                        f"• Bars: {item['bars_seen']} | signals: {item['signal_bars']} "
                        f"| no signal: {item['no_signal_bars']}\n"
                        f"• Skipped: {item['skipped']} | failed: {item['order_failures']} "
                        f"| open: {item['open_trades']}\n"
                        f"• Closed: {closed} | win rate: {win_rate:.1f}% "
                        f"| net: ${item['net_usd']:+.2f} | expectancy: {expectancy_text}"
                    )
                if direction_summaries:
                    lines.append("\n<b>Direction split (realized)</b>")
                    for item in direction_summaries:
                        direction = "BUY" if item["direction"] == 1 else "SELL"
                        closed = int(item["closed_trades"])
                        wins = int(item["wins"] or 0)
                        expectancy = item["expectancy_usd"]
                        expectancy_text = f"${expectancy:+.3f}" if expectancy is not None else "n/a"
                        lines.append(
                            f"• {item['strategy']} {direction}: {wins}/{closed} wins "
                            f"| net ${item['net_usd']:+.2f} | avg {expectancy_text}"
                        )
                if recent:
                    lines.append("\n<b>Recent closed trades</b>")
                    for trade in recent:
                        direction = "BUY" if trade["direction"] == 1 else "SELL"
                        lines.append(
                            f"• {trade['symbol']} {trade['strategy']} {direction}: "
                            f"${trade['realized_net_usd']:+.2f} ({trade['exit_reason']})"
                        )
                bot.reply_to(message, "\n".join(lines))
            except Exception as exc:
                bot.reply_to(message, f"⚠️ Error reading small-profit history: {exc}")

        # --- /analyze ---
        @bot.message_handler(commands=["analyze"])
        def handle_analyze(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            args = message.text.split()
            target = "XAUUSDm"
            if len(args) > 1 and "btc" in args[1].lower():
                target = "BTCUSDm"

            bot.reply_to(message, f"🔍 <i>Running AI Quantitative Engine on {target}...</i>")

            try:
                rates_low = mt5.copy_rates_from_pos(target, mt5.TIMEFRAME_M5, 0, 300)
                rates_high = mt5.copy_rates_from_pos(target, mt5.TIMEFRAME_H1, 0, 400)
                from ai_features import prepare_indicators, extract_single_feature_vector
                import pandas as pd

                df = prepare_indicators(pd.DataFrame(rates_low), pd.DataFrame(rates_high))
                last_c = df.iloc[-1]
                prev_c = df.iloc[-2]

                model_file = "gold_ai_model.pkl" if "XAU" in target else "btc_ai_model.pkl"
                thresh_key = "gold_threshold" if "XAU" in target else "btc_threshold"
                thresh = self.settings.get(thresh_key, 0.65)

                pkg = joblib.load(model_file)
                model = pkg["model"]

                trend = "Bullish 📈" if last_c["h1_trend"] == 1 else ("Bearish 📉" if last_c["h1_trend"] == -1 else "Neutral ⚖️")
                direction = 1 if last_c["h1_trend"] == 1 else -1

                feat = extract_single_feature_vector(last_c, prev_c, direction)
                prob = float(model.predict_proba(feat.reshape(1, -1))[0][1])

                assessment = "High Win-Rate setup criteria met! Waiting for trigger." if prob >= thresh else f"Setup probability ({prob*100:.1f}%) below threshold ({thresh*100:.0f}%). Monitoring market."

                report = (
                    f"🧠 <b>AI INTELLIGENCE REPORT: {target}</b>\n\n"
                    f"• <b>Current Price:</b> ${last_c['close']:,.2f}\n"
                    f"• <b>H1 Trend:</b> {trend} (ADX: {last_c['h1_adx']:.1f})\n"
                    f"• <b>M5 RSI:</b> {last_c['m5_rsi']:.1f}\n"
                    f"• <b>M5 ATR:</b> ${last_c['m5_atr']:.2f}\n"
                    f"• <b>Relative Volume:</b> {last_c.get('m5_vol_ratio', 1.0):.2f}x\n"
                    f"• <b>AI Win Probability:</b> <b>{prob*100:.1f}%</b> (Configured Threshold: {thresh*100:.0f}%)\n\n"
                    f"<b>Assessment:</b> {assessment}"
                )
                bot.reply_to(message, report)

            except Exception as e:
                bot.reply_to(message, f"⚠️ Error running analysis: {e}")

        # --- /pause & /resume ---
        @bot.message_handler(commands=["pause"])
        def handle_pause(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot pause the bot.\nAsk the Bot Creator (GOD) for permission.")
                return
            self.is_paused = True
            self.config["is_paused"] = True
            save_config(self.config)
            bot.reply_to(message, "⏸️ <b>Trading PAUSED.</b> The bot will not enter new positions until /resume is sent.")

        @bot.message_handler(commands=["resume"])
        def handle_resume(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot resume the bot.\nAsk the Bot Creator (GOD) for permission.")
                return
            self.is_paused = False
            self.config["is_paused"] = False
            save_config(self.config)
            bot.reply_to(message, "▶️ <b>Trading RESUMED.</b> The bot is actively scanning for new AI trade setups.")

        # --- /closeall ---
        @bot.message_handler(commands=["closeall"])
        def handle_closeall(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return
            if not self.can_control(cid):
                bot.reply_to(message, "👁️ <b>Read-Only Mode:</b> You cannot close open trades.\nAsk the Bot Creator (GOD) for permission.")
                return

            positions = mt5.positions_get()
            if not positions:
                bot.reply_to(message, "ℹ️ No open positions to close.")
                return

            closed_count = 0
            for pos in positions:
                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(pos.symbol)
                price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 30,
                    "comment": "Telegram_Manual_CloseAll"
                }
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    closed_count += 1

            bot.reply_to(message, f"🚨 <b>Emergency Close Complete!</b> Closed {closed_count} of {len(positions)} positions.")

        # --- User & Permission Management (GOD only) ---
        @bot.message_handler(commands=["adduser"])
        def handle_adduser(message):
            cid = message.chat.id
            if not self.is_god(cid):
                bot.reply_to(message, "👑 <b>Access Denied:</b> Only the Bot Creator (GOD) can authorize new users.")
                return

            parts = message.text.split()
            if len(parts) < 2 or not parts[1].isdigit():
                bot.reply_to(
                    message,
                    "⚠️ Usage: <code>/adduser &lt;chat_id&gt; [admin|viewer]</code>\n"
                    "Example:\n"
                    "• <code>/adduser 987654321</code> (Defaults to 👁️ Read-Only Viewer)\n"
                    "• <code>/adduser 987654321 admin</code> (Grants 🛠️ Full Admin Control)"
                )
                return

            new_id = int(parts[1])
            assigned_role = ROLE_ADMIN if len(parts) >= 3 and parts[2].lower() == "admin" else ROLE_VIEWER

            self.authorized_chat_ids.add(new_id)
            self.user_roles[str(new_id)] = assigned_role
            self.config["authorized_chat_ids"] = list(self.authorized_chat_ids)
            self.config["user_roles"] = self.user_roles
            save_config(self.config)

            role_badge = "🛠️ <b>ADMIN</b> (Full Trading & Settings Control)" if assigned_role == ROLE_ADMIN else "👁️ <b>VIEWER</b> (Read-Only: Telemetry & AI Chat only)"
            bot.reply_to(
                message,
                f"✅ User <code>{new_id}</code> authorized!\n\n"
                f"• <b>Role:</b> {role_badge}\n"
                f"<i>(You can change their permission anytime with: <code>/grant {new_id} admin</code> or <code>/grant {new_id} viewer</code>)</i>"
            )

        @bot.message_handler(commands=["grant"])
        def handle_grant(message):
            cid = message.chat.id
            if not self.is_god(cid):
                bot.reply_to(message, "👑 <b>Access Denied:</b> Only the Bot Creator (GOD) can grant permissions.")
                return

            parts = message.text.split()
            if len(parts) < 3 or not parts[1].isdigit():
                bot.reply_to(
                    message,
                    "⚠️ Usage: <code>/grant &lt;chat_id&gt; &lt;admin|viewer&gt;</code>\n\n"
                    "• <code>/grant &lt;id&gt; admin</code> — Give full control (execute trades, modify risk/settings)\n"
                    "• <code>/grant &lt;id&gt; viewer</code> — Set to read-only (telemetry & AI chat only)"
                )
                return

            target_id = int(parts[1])
            new_role = parts[2].lower()

            if new_role not in [ROLE_ADMIN, ROLE_VIEWER]:
                bot.reply_to(message, "⚠️ Invalid role! Specify either <code>admin</code> or <code>viewer</code>.")
                return

            if target_id == self.god_id:
                bot.reply_to(message, "👑 The GOD account cannot be demoted or modified!")
                return

            self.authorized_chat_ids.add(target_id)
            self.user_roles[str(target_id)] = new_role
            self.config["authorized_chat_ids"] = list(self.authorized_chat_ids)
            self.config["user_roles"] = self.user_roles
            save_config(self.config)

            role_badge = "🛠️ <b>ADMIN (Full Control)</b>" if new_role == ROLE_ADMIN else "👁️ <b>VIEWER (Read-Only)</b>"
            bot.reply_to(
                message,
                f"🎉 <b>Permissions Updated:</b> User <code>{target_id}</code> is now {role_badge}!"
            )

            # Notify the user
            try:
                bot.send_message(
                    target_id,
                    f"🔔 <b>Your bot permissions were updated by the GOD!</b>\n"
                    f"Your new access level is: {role_badge}"
                )
            except Exception:
                pass

        @bot.message_handler(commands=["removeuser"])
        def handle_removeuser(message):
            cid = message.chat.id
            if not self.is_god(cid):
                bot.reply_to(message, "👑 <b>Access Denied:</b> Only the Bot Creator (GOD) can remove users.")
                return

            parts = message.text.split()
            if len(parts) < 2 or not parts[1].isdigit():
                bot.reply_to(message, "⚠️ Usage: <code>/removeuser &lt;chat_id&gt;</code>")
                return

            target_id = int(parts[1])
            if target_id == self.god_id:
                bot.reply_to(message, "👑 The GOD account cannot be removed!")
                return

            if target_id in self.authorized_chat_ids:
                self.authorized_chat_ids.remove(target_id)
                self.config["authorized_chat_ids"] = list(self.authorized_chat_ids)
                if str(target_id) in self.user_roles:
                    del self.user_roles[str(target_id)]
                self.config["user_roles"] = self.user_roles
                save_config(self.config)
                bot.reply_to(message, f"🗑️ User <code>{target_id}</code> removed from authorized list.")
            else:
                bot.reply_to(message, f"ℹ️ User <code>{target_id}</code> was not in the authorized list.")

        @bot.message_handler(commands=["listusers"])
        def handle_listusers(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            lines = []
            for uid in sorted(self.authorized_chat_ids):
                r = self.get_user_role(uid)
                if r == ROLE_GOD:
                    badge = "👑 <b>GOD / Creator</b> (Permanent Master)"
                elif r == ROLE_ADMIN:
                    badge = "🛠️ <b>Admin</b> (Full Control)"
                else:
                    badge = "👁️ <b>Viewer</b> (Read-Only)"
                lines.append(f"• <code>{uid}</code> — {badge}")

            bot.reply_to(
                message,
                f"👥 <b>AUTHORIZED USERS & PERMISSIONS ({len(self.authorized_chat_ids)})</b>\n\n"
                + "\n".join(lines)
                + "\n\n<i>To change permissions: <code>/grant &lt;id&gt; admin</code> or <code>/grant &lt;id&gt; viewer</code></i>"
            )

        # --- Button Text Handler & Natural Chat ---
        @bot.message_handler(func=lambda msg: True)
        def handle_button_clicks_and_chat(message):
            cid = message.chat.id
            if not self.is_authorized(cid):
                return

            text = message.text.strip()

            # Handle interactive awaiting states (e.g. typing a new API key)
            if cid in self.awaiting_input:
                mode = self.awaiting_input[cid]
                if text.lower() in ["/cancel", "cancel"]:
                    del self.awaiting_input[cid]
                    bot.reply_to(message, "❌ Operation cancelled.")
                    return

                if mode == "gemini_key":
                    new_key = text.strip()
                    try:
                        bot.delete_message(cid, message.message_id)
                    except Exception:
                        pass

                    bot.send_chat_action(cid, "typing")
                    test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={new_key}"
                    try:
                        r = requests.post(test_url, json={"contents": [{"parts": [{"text": "ping"}]}]}, timeout=10)
                        if r.status_code == 200:
                            self.config["gemini_api_key"] = new_key
                            save_config(self.config)
                            del self.awaiting_input[cid]
                            bot.send_message(
                                cid,
                                f"✅ <b>Gemini API Key Verified & Updated!</b>\n\n"
                                f"• Key: <code>{new_key[:8]}...{new_key[-4:]}</code>\n"
                                f"• Model: <code>gemini-3.1-flash-lite</code>\n"
                                f"• Status: Active & Ready! 🎉\n"
                                f"<i>(Your API key message was auto-deleted for privacy)</i>"
                            )
                        else:
                            bot.send_message(cid, f"❌ <b>Key Validation Failed (HTTP {r.status_code}):</b>\n{r.text[:200]}\n\nTry sending the key again, or send <code>/cancel</code>.")
                    except Exception as e:
                        bot.send_message(cid, f"⚠️ Error testing key: {e}\n\nTry again or send <code>/cancel</code>.")
                    return

            # Handle Quick Signal Button
            if text == "🎯 Paste Signal":
                handle_signal_command(message)
                return

            # Check if user pasted a trade signal in regular chat
            sig_data = self.parse_trading_signal(text)
            if sig_data:
                self.prompt_signal_confirmation(message, sig_data)
                return

            # Handle quick buttons
            if text == "📊 Status":
                handle_status(message)
            elif text == "💰 Balance":
                handle_balance(message)
            elif text == "📋 Positions":
                handle_positions(message)
            elif text == "👤 Account":
                handle_account(message)
            elif text == "⚙️ Settings":
                handle_settings(message)
            elif text == "📊 Token Usage":
                handle_tokens(message)
            elif text == "🧠 Analyze Gold":
                message.text = "/analyze gold"
                handle_analyze(message)
            elif text == "🧠 Analyze BTC":
                message.text = "/analyze btc"
                handle_analyze(message)
            elif text == "⏸️ Pause/Resume":
                if self.is_paused:
                    handle_resume(message)
                else:
                    handle_pause(message)
            elif text == "🚨 Close All":
                handle_closeall(message)
            elif text == "🔄 Restart Bot":
                handle_restart(message)
            elif text in ["📰 News", "📰 Economic News"]:
                self.send_news_summary(message)
            else:
                self._handle_conversational_chat(message, text)

    def _handle_conversational_chat(self, message, user_query):
        bot = self.bot
        cid = message.chat.id

        acc = mt5.account_info()
        positions = mt5.positions_get() or []
        gold_tick = mt5.symbol_info_tick("XAUUSDm")
        btc_tick = mt5.symbol_info_tick("BTCUSDm")

        gold_price = gold_tick.bid if gold_tick else 0.0
        btc_price = btc_tick.bid if btc_tick else 0.0

        # Technical telemetry snapshot for AI reasoning
        gold_intel = "Unavailable"
        btc_intel = "Unavailable"
        try:
            from ai_features import prepare_indicators
            import pandas as pd
            g_low = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M5, 0, 100)
            g_high = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_H1, 0, 400)
            if g_low is not None and g_high is not None and len(g_high) >= 200:
                gdf = prepare_indicators(pd.DataFrame(g_low), pd.DataFrame(g_high))
                gl = gdf.iloc[-1]
                gt = "Bullish 📈" if gl["h1_trend"] == 1 else ("Bearish 📉" if gl["h1_trend"] == -1 else "Neutral ⚖️")
                gold_intel = f"${gold_price:,.2f} | H1 Trend: {gt} (ADX: {gl['h1_adx']:.1f}) | M5 RSI: {gl['m5_rsi']:.1f} | ATR: ${gl['m5_atr']:.2f}"
        except Exception:
            pass

        try:
            from ai_features import prepare_indicators
            import pandas as pd
            b_low = mt5.copy_rates_from_pos("BTCUSDm", mt5.TIMEFRAME_M5, 0, 100)
            b_high = mt5.copy_rates_from_pos("BTCUSDm", mt5.TIMEFRAME_H1, 0, 400)
            if b_low is not None and b_high is not None and len(b_high) >= 200:
                bdf = prepare_indicators(pd.DataFrame(b_low), pd.DataFrame(b_high))
                bl = bdf.iloc[-1]
                bt = "Bullish 📈" if bl["h1_trend"] == 1 else ("Bearish 📉" if bl["h1_trend"] == -1 else "Neutral ⚖️")
                btc_intel = f"${btc_price:,.2f} | H1 Trend: {bt} (ADX: {bl['h1_adx']:.1f}) | M5 RSI: {bl['m5_rsi']:.1f} | ATR: ${bl['m5_atr']:.2f}"
        except Exception:
            pass

        gemini_key = self.config.get("gemini_api_key", "").strip()
        if gemini_key:
            try:
                bot.send_chat_action(cid, "typing")
                pos_summary = ", ".join([f"#{p.ticket} {p.symbol} {'BUY' if p.type==0 else 'SELL'} {p.volume}L @ ${p.price_open:.2f} (PnL: ${p.profit:.2f})" for p in positions]) or "None"
                system_prompt = (
                    "You are PhantomRex AI, an elite institutional quantitative trading assistant for MetaTrader 5.\n"
                    "LIVE REAL-TIME TELEMETRY:\n"
                    f"- Account Balance: ${acc.balance if acc else 0:,.2f} | Equity: ${acc.equity if acc else 0:,.2f} | Margin Level: {acc.margin_level if acc else 0:.1f}%\n"
                    f"- Gold (XAUUSDm): {gold_intel}\n"
                    f"- Bitcoin (BTCUSDm): {btc_intel}\n"
                    f"- Active Open Trades: {pos_summary}\n"
                    f"- Bot Status: {'PAUSED' if self.is_paused else 'ACTIVE'}\n"
                    f"- Target Risk: {self.settings.get('risk_percent', 1.5)}% | AI Confidence Cutoff: {self.settings.get('gold_threshold', 0.65)*100:.0f}%\n\n"
                    "CRITICAL RESPONSE GUIDELINES:\n"
                    "1. BE CONCISE AND PUNCHY: Maximum 2 to 3 short sentences or bullet points. Never write long essays or rambling paragraphs.\n"
                    "2. HIGH-IQ QUANT TONE: Answer directly with quantitative facts, price levels, and indicator readings.\n"
                    "3. NO FLUFF: Skip introductory pleasantries and disclaimers. Get straight to the answer."
                )

                for model_candidate in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_candidate}:generateContent?key={gemini_key}"
                    payload = {
                        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser: {user_query}"}]}],
                        "generationConfig": {
                            "maxOutputTokens": 200,
                            "temperature": 0.3
                        }
                    }
                    res = requests.post(url, json=payload, timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        ans = data["candidates"][0]["content"]["parts"][0]["text"]

                        meta = data.get("usageMetadata", {})
                        p_tokens = meta.get("promptTokenCount", 0)
                        c_tokens = meta.get("candidatesTokenCount", 0)
                        self._record_token_usage(p_tokens, c_tokens)

                        formatted_ans = format_telegram_response(ans)
                        bot.reply_to(message, formatted_ans)
                        return
            except Exception as e:
                print(f"⚠️ Gemini API query failed: {e}")

        # Streamlined Financial Reasoning Fallback
        q = user_query.lower()

        if any(w in q for w in ["thank", "thanks", "appreciate"]):
            bot.reply_to(message, "Always on duty. 📈 Let me know what you need.")
            return
        if any(w in q for w in ["hello", "hi", "hey"]):
            bot.reply_to(message, f"Online. ⚡ Gold: ${gold_price:,.2f} | BTC: ${btc_price:,.2f}. How can I assist?")
            return

        if any(w in q for w in ["news", "economic", "calendar", "cpi", "nfp", "fomc"]):
            self.send_news_summary(message)
            return

        if "gold" in q or "xau" in q:
            bot.reply_to(
                message,
                f"🥇 <b>Gold (XAUUSDm):</b> ${gold_price:,.2f}\n"
                f"• {gold_intel}\n"
                f"• Scanner: Watching for M5 pullback triggers aligned with macro structure."
            )
            return

        if "btc" in q or "bitcoin" in q or "crypto" in q:
            bot.reply_to(
                message,
                f"₿ <b>Bitcoin (BTCUSDm):</b> ${btc_price:,.2f}\n"
                f"• {btc_intel}\n"
                f"• Scanner: 24/7 crypto momentum active."
            )
            return

        if ("why" in q and ("trade" in q or "enter" in q or "buy" in q or "sell" in q)) or "waiting" in q:
            if self.is_paused:
                status_note = "Bot is currently PAUSED (/resume to activate)."
            elif len(positions) >= self.settings.get("max_positions", 6):
                status_note = "Max concurrent positions reached."
            else:
                thresh = self.settings.get('gold_threshold', 0.65)*100
                status_note = f"Market is between setups. Scanner is waiting for a pullback meeting >{thresh:.0f}% AI conviction."

            bot.reply_to(
                message,
                f"🎯 <b>Execution Status:</b>\n"
                f"• {status_note}\n"
                f"• Capital is preserved until high-probability confluence aligns."
            )
            return

        if "profit" in q or "balance" in q or "money" in q or "equity" in q:
            if acc:
                floating = acc.equity - acc.balance
                fl_str = f"+${floating:,.2f} 🟢" if floating >= 0 else f"-${abs(floating):,.2f} 🔴"
                bot.reply_to(
                    message,
                    f"💰 <b>Account Health:</b>\n"
                    f"• Balance: <b>${acc.balance:,.2f}</b> | Equity: <b>${acc.equity:,.2f}</b>\n"
                    f"• Floating PnL: {fl_str} | Open Trades: {len(positions)}"
                )
                return

        if "position" in q or "trade" in q or "order" in q:
            if not positions:
                bot.reply_to(message, "ℹ️ <b>0 active positions.</b> Capital is 100% in cash.")
            else:
                lines = [f"📋 <b>Active Positions ({len(positions)}):</b>\n"]
                for p in positions:
                    lines.append(f"• #{p.ticket} {p.symbol} ({'BUY' if p.type==0 else 'SELL'}) {p.volume}L | PnL: <b>${p.profit:+.2f}</b>")
                bot.reply_to(message, "\n".join(lines))
            return

        bot.reply_to(
            message,
            f"🤖 <b>PhantomRex Telemetry:</b>\n\n"
            f"• <b>Gold (XAUUSDm):</b> {gold_intel}\n"
            f"• <b>Bitcoin (BTCUSDm):</b> {btc_intel}\n"
            f"• <b>Open Trades:</b> {len(positions)} active | <b>Balance:</b> ${acc.balance if acc else 0:,.2f}\n\n"
            f"💡 <i>Tip: Send /menu for controls, /signal to paste trade signals, or ask me any market question!</i>"
        )
