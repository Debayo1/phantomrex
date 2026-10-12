# PRD: AIBot\_TP\_Core v2

**Owner:** Oluwatosin **Status:** Draft **Scope:** BTCUSDm and XAUUSDm on MetaTrader 5, with Telegram control and alerts

## 1. Overview

AIBot\_TP\_Core is a self-built AI trading bot that opens sell/buy trades on BTCUSDm and XAUUSDm, uses a model confidence score to decide whether to trade, moves the stop loss to breakeven once a trade is going well, and reports through Telegram. The bot is live at 0.01 lots with no closed losses so far. This version focuses on proving the edge is real, fixing risk reporting, and adding guardrails before the lot size is increased.

## 2. Problem Statement

- The retrain report shows a \~96% historical win rate, but live confidence is \~68% and the recommended cutoff is 65%. The numbers don't agree, which suggests the backtest is inflated.
- It is unclear how "win" is defined in training. If it means "price moved far enough to trigger breakeven," a high win rate can still hide negative expectancy.
- The initial stop (about 151 points) is far larger than the breakeven trigger (about 15 points), so one early reversal costs roughly ten small wins.
- The Telegram alert reports total risk of $0.16 for a \~151-point stop at 0.01 lots, which looks about 10x too low and may point to a contract-size or tick-value bug.
- There are not yet enough closed trades to judge live performance.

## 3. Goals

1. Replace the current validation with a leak-free, time-aware process whose results can be trusted.
2. Measure real expectancy (including spread, slippage and the breakeven logic), not just win rate.
3. Make risk numbers in alerts and sizing correct.
4. Log every signal and trade so confidence can be checked against actual outcomes.
5. Add hard risk limits so a bad run cannot cause a large loss.

## 4. Non-Goals

- Increasing lot size or adding new instruments in this version.
- Building a mobile app or web dashboard (Telegram stays the interface).
- Changing the core model type unless validation shows it is necessary.

## 5. Users

Single user: the bot owner, monitoring and controlling the bot from Telegram while it runs unattended on MT5.

## 6. Current State

| Area | Today |
| --- | --- |
| Instruments | BTCUSDm, XAUUSDm |
| Lot size | 0.01 (HIGH CONVICTION tier = 1.00x) |
| Entry rule | Model confidence above cutoff (currently 65%) |
| Stop loss | Fixed initial distance (\~151.55 on a BTC sell) |
| Stop management | Auto-breakeven, then trails to lock profit |
| Training | /retrain pulls 40,000 candles, 5-fold CV, calibrates cutoff |
| Reporting | Telegram alerts for new trades and breakeven locks |

## 7. Requirements

### 7.1 Trade-Outcome Labeling (P0)

- Label each setup by simulating the full trade: initial stop, breakeven trigger, trailing logic, and take profit.
- Record the outcome in account currency (profit or loss), not just win/loss.
- Report win rate, average win, average loss, and expectancy per instrument.

### 7.2 Time-Aware Validation (P0)

- Replace random 5-fold CV with walk-forward validation: train on older data, test only on later data.
- Add a purge/embargo gap equal to the maximum trade duration so overlapping setups do not leak between train and test.
- Report metrics per fold and in aggregate, plus the worst fold.

### 7.3 Leakage Audit (P0)

- Review every feature to confirm it only uses data available at the moment of the signal (no use of the still-forming candle's close).
- Add a test that shifts features by one candle and confirms performance drops as expected.

### 7.4 Cost Modelling (P0)

- Include the broker's real spread (time-varying if available) and a slippage allowance in all backtests.
- Show results before and after costs.

### 7.5 Risk Calculation Fix (P0)

- Verify contract size and tick value for each symbol against MT5 symbol info.
- Correct the "Total Risk Allocated" calculation so it equals (stop distance × lots × contract size) in account currency.
- Add a unit test comparing the calculated risk against MT5's own order-check or margin calculations.

### 7.6 Signal and Trade Logging (P1)

- Log every signal (taken or skipped) with timestamp, symbol, features, confidence, entry, stop, take profit, and spread.
- Log every closed trade with outcome, duration, whether breakeven triggered, and final profit or loss.
- Store logs in a database or CSV that survives restarts.

### 7.7 Calibration Reporting (P1)

- Add a Telegram command (for example /stats) that returns closed trade count, win rate, average win, average loss, expectancy, max drawdown, and current floating drawdown.
- Add a calibration table comparing confidence buckets (60-65%, 65-70%, and so on) to actual win rates.

### 7.8 Risk Guardrails (P0)

- Maximum open positions per symbol and in total.
- Daily loss limit and maximum drawdown limit, after which the bot stops opening trades and sends an alert.
- Disallow adding to losing positions unless explicitly enabled.
- A /pause and /resume command, and a kill switch that closes all positions.

### 7.9 Retrain Safety (P1)

- After /retrain, compare the new model against the current one on held-out later data and only hot-reload if it is not worse.
- Keep the previous model for one-command rollback.
- Report out-of-sample metrics in the retrain message instead of in-sample win rate.

## 8. Success Metrics

| Metric | Target |
| --- | --- |
| Out-of-sample expectancy per trade (after costs) | Positive in at least 4 of 5 walk-forward folds |
| Risk shown in alerts vs. actual stop-out loss | Within 2% |
| Live closed trades before any lot increase | 50 or more |
| Live win rate vs. confidence-implied win rate | Within 10 percentage points |
| Max drawdown | Within the limit set in 7.8 |

## 9. Rollout Plan

1. **Phase 1, offline:** build the labeling, walk-forward validation, leakage audit and cost model. Decide whether the edge survives.
2. **Phase 2, fixes:** correct the risk calculation and add logging, /stats and guardrails.
3. **Phase 3, demo or 0.01 live:** run for at least 2 to 4 weeks until 50 or more closed trades.
4. **Phase 4, review:** compare live results to backtest. Only then consider a gradual lot increase.

## 10. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Honest validation shows no real edge | Pause live trading, rework features or labels before continuing |
| Rare large loss from the wide initial stop | Daily loss limit, drawdown limit, tighter stop experiments tested offline |
| Bot or MT5 connection drops with positions open | Heartbeat alert and server-side stops on every trade |
| Model drift after retraining | Out-of-sample comparison and rollback (7.9) |
| Small sample misleads decisions | No lot increase before 50+ closed trades |

## 11. Open Questions

- How exactly is a "win" defined in the current training labels?
- What is the maximum trade duration, to set the purge gap?
- Is the account a standard or cent account, and what are the contract sizes?
- What daily loss and drawdown limits are acceptable?
- Does the bot ever open multiple trades in the same direction at once?
