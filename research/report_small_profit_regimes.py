"""Generate a read-only markdown report for demo small-profit context."""

import os

import small_profit_ledger


REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runtime_data",
    "small_profit_regime_report.md",
)


def render_regime_report(summary=None, regime_rows=None, shadow=None):
    summary = summary or []
    regime_rows = regime_rows or []
    shadow = shadow or {}
    lines = [
        "# Small-Profit Strategy / Regime Report",
        "",
        "**Scope:** demo-only ledger observations. Context labels are descriptive and are not live routing decisions.",
        "",
        "## Strategy totals",
        "",
        "| Strategy | Bars | Signals | Skipped | Closed | Wins | Net USD | Expectancy USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        expectancy = row.get("expectancy_usd")
        expectancy_text = "n/a" if expectancy is None else f"${expectancy:+.3f}"
        lines.append(
            f"| {row['strategy']} | {row['bars_seen']} | {row['signal_bars']} | "
            f"{row['skipped']} | {row['closed_trades']} | {row['wins']} | "
            f"${row['net_usd']:+.2f} | {expectancy_text} |"
        )

    lines += [
        "",
        "## Closed results by market context",
        "",
        "Cells with fewer than 30 trades are observations, not promotion evidence.",
        "",
        "| Symbol | Strategy | Regime | Session | Volatility | Trades | Wins | Win rate | Net USD | Expectancy USD |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(regime_rows, key=lambda item: (
        item["symbol"], item["strategy"], item["regime"], item["session"]
    )):
        closed = int(row["closed_trades"])
        wins = int(row["wins"])
        expectancy = row.get("expectancy_usd")
        expectancy_text = "n/a" if expectancy is None else f"${expectancy:+.3f}"
        lines.append(
            f"| {row['symbol']} | {row['strategy']} | {row['regime']} | {row['session']} | "
            f"{row['volatility_bucket']} | {closed} | {wins} | "
            f"{wins / closed * 100:.1f}% | ${row['net_usd']:+.2f} | {expectancy_text} |"
        )

    lines += [
        "",
        "## Interpretive limits",
        "",
        "- Positive totals do not prove a durable edge.",
        "- Net USD includes realized deal profit, commission, swap, and fee fields available in MT5.",
        "- Context fields describe market state and must be tested chronologically before becoming execution rules.",
        "- No pair, lot, live-mode, or model-promotion change is authorized by this report.",
        "",
    ]
    if shadow:
        lines += [
            "## Existing 00:00 UTC XAU M1-breakout shadow filter",
            "",
            f"Rule: `{shadow.get('rule_id', 'n/a')}`",
            "",
            f"Shadow candidates: {shadow.get('shadow_candidates', 0)}",
            "",
        ]
    return "\n".join(lines)


def run_report(db_path=None):
    summary = small_profit_ledger.get_summary(db_path) if db_path else small_profit_ledger.get_summary()
    regime_rows = small_profit_ledger.get_regime_summary(db_path) if db_path else small_profit_ledger.get_regime_summary()
    shadow = small_profit_ledger.get_shadow_filter_summary(db_path) if db_path else small_profit_ledger.get_shadow_filter_summary()
    report = render_regime_report(summary, regime_rows, shadow)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as file:
        file.write(report)
    return REPORT_PATH


if __name__ == "__main__":
    print(run_report())
