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
