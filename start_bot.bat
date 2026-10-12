@echo off
title Dual-Asset AI Trading Bot (XAUUSDm & BTCUSDm) - 24/7 AWS
cd /d %~dp0

echo ======================================================================
echo          PHANTOMREX AI QUANTITATIVE TRADING BOT (MT5 + TELEGRAM)
echo ======================================================================
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not found in system PATH.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies...
python -c "import telebot, MetaTrader5, pandas, numpy, joblib" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Missing dependencies detected. Installing requirements.txt...
    pip install -r requirements.txt
)

echo [3/3] Launching AI Trading Engine & Telegram Remote Controller...
echo ======================================================================
echo.

:loop
python bot.py
echo.
echo ======================================================================
echo [ALERT] Bot process exited at %date% %time%
echo Restarting automatically in 10 seconds (Press Ctrl+C to stop)...
echo ======================================================================
timeout /t 10 /nobreak
goto loop

