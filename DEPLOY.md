# Deployment & VPS Update Guide

## ⚠️ Critical Step Before First `git pull` on AWS VPS (Addendum D1)

Because `telegram_config.json` is being untracked from Git for security, pulling this update on your AWS VPS will cause Git to delete the untracked file from your working tree.

### Follow this procedure on your AWS VPS:

#### 1. Back Up Your Config File First
Before running `git pull`, copy your existing `telegram_config.json` to a safe location outside the repo folder:
```cmd
copy "C:\Users\Administrator\Desktop\forexbot\telegram_config.json" "C:\Users\Administrator\Desktop\telegram_config.backup.json"
```

#### 2. Pull the Latest Code
```cmd
cd "C:\Users\Administrator\Desktop\forexbot"
git pull
```

#### 3. Restore the Config File
Restore your backed-up configuration back into the repo folder (or migrate secrets into a `.env` file):
```cmd
copy "C:\Users\Administrator\Desktop\telegram_config.backup.json" "C:\Users\Administrator\Desktop\forexbot\telegram_config.json"
```

#### 4. Verify Bot Startup
Run the bot and verify that credentials load properly without placeholder warnings:
```cmd
python bot.py
```
Check that the bot connects to MT5 and starts polling Telegram before leaving the console.

#### 5. Strict Graduation & Account Safety
Before switching to real money trading, consult [GRADUATION.md](file:///c:/Users/HP%201030/Desktop/forexbot/GRADUATION.md) for the mandatory 50-trade demo evaluation gate and 0.01-lot size constraints.

## Runtime Research Logs

Research and retraining runs append to CSV files under `runtime_data/`, which is ignored by Git so VPS-generated results do not block or get overwritten by `git pull`. The tracked `research/registry.csv` is a historical seed/reference file and is no longer the destination for new runtime rows.

Back up `runtime_data/` outside the repository before replacing or rebuilding the VPS checkout. Do not copy runtime logs back over the tracked seed registry. If migrating older VPS rows from a Git stash, preserve a backup and append/import those rows into the appropriate runtime log after updating; strategy and retraining logs use separate CSV schemas.

## V7 Demo Forward Test

The 9am CR (V7) strategy is enabled only when the connected MT5 account reports demo trade mode. On demo accounts it replaces the bot's existing M5/H1 entry strategy; real accounts continue using the existing strategy and cannot open V7 positions. V7 uses the New York 08:00 range and 09:00-10:00 M1 setup, the existing daily-loss/news/position guards, the configured risk percentage and 0.01-lot cap, and a 16:00 New York time exit. It does not require the AI model edge gate. Treat it as a demo forward test, not as evidence of profitability or approval for live trading.

## V8 SMC Forward Paper Test

V8 is research-only and never sends broker orders. Run it in a separate terminal while MT5 is connected:

```cmd
python -m research.paper_logger_v8 60
```

The first run sets the current M15 baseline and does not backfill old trades. Keep the process running; it evaluates new M15 bars and records simulated opens and closes to `runtime_data/v8_forward_paper.csv`, with restart state in `runtime_data/v8_forward_state.json`. Preserve both files when updating or restarting the VPS. Do not delete the state file while monitoring, or the next startup will establish a new baseline. Results are a forward paper sample, not proof of profitability or permission to enable V8 orders.
