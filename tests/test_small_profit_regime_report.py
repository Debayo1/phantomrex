from research.report_small_profit_regimes import render_regime_report


def test_regime_report_discloses_sample_limitations():
    report = render_regime_report(
        summary=[{
            "strategy": "Trend pullback",
            "bars_seen": 10,
            "signal_bars": 5,
            "skipped": 2,
            "closed_trades": 3,
            "wins": 2,
            "net_usd": 0.75,
            "expectancy_usd": 0.25,
        }],
        regime_rows=[{
            "symbol": "XAUUSDm",
            "strategy": "Trend pullback",
            "regime": "strong_uptrend",
            "session": "london_new_york_overlap",
            "volatility_bucket": "normal",
            "closed_trades": 3,
            "wins": 2,
            "net_usd": 0.75,
            "expectancy_usd": 0.25,
        }],
        shadow={"rule_id": "test_rule", "shadow_candidates": 4},
    )

    assert "Cells with fewer than 30 trades" in report
    assert "XAUUSDm | Trend pullback | strong_uptrend" in report
    assert "No pair, lot, live-mode, or model-promotion change" in report


def test_empty_report_is_still_a_valid_markdown_report():
    report = render_regime_report()

    assert report.startswith("# Small-Profit Strategy / Regime Report")
    assert "## Strategy totals" in report
