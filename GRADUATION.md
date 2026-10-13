# Institutional Trading Graduation & Go-Live Protocol

This document defines the strict, institutional operational rules governing account promotion, lot size scaling, and risk controls for the PhantomRex MT5 trading bot (`XAUUSDm` and `BTCUSDm`).

---

## 🛡️ The 6 Non-Negotiable Operational Rules

### 1. Demo First Deployment
Upon deployment or update to the AWS VPS, the bot must initially run strictly on the **Demo account**:
* In `telegram_config.json` (or via `/login demo`), `active_account` must be `"demo"`.
* Live trading with real capital is prohibited until all graduation requirements below are met.

### 2. 0.01 Lots Exclusively
* The bot trades strictly with the minimum broker volume: **0.01 lots**.
* This is enforced at the code level by the hard constant:
  ```python
  # ONLY OWNER MAY CHANGE THIS CONSTANT
  MAX_LOT_CAP = 0.01
  FORCE_SINGLE_POSITION = True
  ```
* Any attempt by settings or formulas to open larger lot sizes or multi-target stacked positions is clamped to `0.01` and a single trade.

---

## 🎓 The 50-Trade Demo Graduation Gate (Addendum D2)

Promotion from Demo to the Real live account is permitted **ONLY** after fulfilling all three criteria over at least **50 closed demo trades executed under the new architecture**:

> [!IMPORTANT]
> Historical demo trades executed prior to the deployment of Phase 1 do **NOT** count toward the 50-trade quota. Only trades tracked in `signals.db` and matched via `deal.position_id == signals.position_ticket` are valid.

### Graduation Criteria:
1. **Positive Net Expectancy:**
   $$\text{Realized Net Expectancy} > +0.00\text{R}$$
   (inclusive of all spreads, commissions, and swap fees).
2. **Confidence Calibration Parity:**
   * Realized win rate matches the model's out-of-sample calibration table within **$\pm 10$ percentage points**.
   * Inspected via `/stats` across populated calibration buckets ($\ge 10$ trades each).
3. **Zero Software Guard Trips:**
   * **Zero** daily loss guard trips caused by software bugs, duplicate fills, slippage anomalies, or orphaned positions.

---

## 💰 Real Account 50-Trade Graduation Gate

Once graduated to the Real account:
1. **Initial Real Size:** Trading begins at **0.01 lots exclusively**.
2. **Evaluation Window:** The real account must complete its own **50 closed real trades**.
3. **Scaling Authorization:** Only after 50 real trades demonstrate positive expectancy ($> 0.00\text{R}$) without risk guard failures may the bot owner (GOD) consider scaling up position sizing.

---

## 👑 Owner Authority & Role Permissions
* **GOD User Exclusivity:** Only the bot owner (`ROLE_GOD`) is authorized to execute:
  * `/login live` (switching accounts)
  * `/rollback` (restoring previous AI models)
  * Modifying risk parameters or adding authorized admins.
* **Viewer Role:** Guest and read-only users cannot execute trades, toggle breakeven, or switch accounts.

---

## 📅 Weekly Review & Model Audit Runbook

Every Sunday at 8:00 PM UTC:
1. **Automated Weekly Retraining:** The bot pulls the freshest 40,000 candles and evaluates candidate models against the Phase 4 fair holdout promotion gate.
2. **Promotion Inspection:**
   * If promoted: Telegram sends `🏆 PROMOTED` alert; models are hot-reloaded into memory and backed up to `<model>.prev.pkl`.
   * If rejected: Incumbent models remain active untouched.
3. **Weekly Telemetry Review:**
   * Run `/stats` in Telegram.
   * Compare realized R-multiples and win rate against the audit reports in `reports/`.
   * If live performance experiences severe edge decay, execute `/rollback` to restore the previous validated model package.
