"""
test_sandbox_advanced.py — Tests the sandbox with:
  1. A backtest that reads CSV data (generates synthetic CSV first)
  2. Produces a plot saved to disk
  3. Returns JSON metrics

Run: python3 test_sandbox_advanced.py

No LLM, no LangGraph, no API keys needed.
"""

import json
import logging
import os
import textwrap

from sandbox_executor import SandboxExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

DIVIDER = "=" * 70


def test_backtest_with_data_and_plot():
    print(f"\n{DIVIDER}")
    print("TEST — Backtest with CSV data input + plot output")
    print(DIVIDER)

    # ── Step 1: The backtest code ────────────────────────────────────
    # This simulates what your Agent 5+6 pipeline would produce:
    #   - Reads price data from a CSV
    #   - Computes a simple moving average crossover strategy
    #   - Generates an equity curve plot
    #   - Prints JSON metrics
    backtest_code = textwrap.dedent("""\
        import json
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        import os

        # ── Load data ────────────────────────────────────────────────
        DATA_PATH = os.path.join(os.path.dirname(__file__), "input_data.csv")

        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        else:
            # Fallback: generate synthetic data if file missing
            np.random.seed(42)
            dates = pd.date_range("2018-01-01", periods=252 * 5, freq="B")
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
            df = pd.DataFrame({"date": dates, "close": prices})

        df = df.sort_values("date").reset_index(drop=True)

        # ── Strategy: SMA crossover ─────────────────────────────────
        df["sma_fast"] = df["close"].rolling(window=20).mean()
        df["sma_slow"] = df["close"].rolling(window=50).mean()
        df["signal"] = np.where(df["sma_fast"] > df["sma_slow"], 1, -1)
        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
        df = df.dropna()

        # ── Metrics ─────────────────────────────────────────────────
        cumulative = (1 + df["strategy_returns"]).cumprod()
        buy_hold = (1 + df["returns"]).cumprod()

        ann_return = df["strategy_returns"].mean() * 252
        ann_vol = df["strategy_returns"].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        max_dd = float((cumulative / cumulative.cummax() - 1).min())
        total_return = float(cumulative.iloc[-1] - 1)
        bh_return = float(buy_hold.iloc[-1] - 1)
        win_rate = float((df["strategy_returns"] > 0).mean())
        n_trades = int((df["signal"].diff().abs() > 0).sum())

        metrics = {
            "sharpe_ratio": round(sharpe, 4),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(total_return, 4),
            "buy_hold_return": round(bh_return, 4),
            "win_rate": round(win_rate, 4),
            "num_trades": n_trades,
            "n_trading_days": len(df),
        }

        # ── Plot: equity curves ─────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top: equity curves
        axes[0].plot(df["date"].values, cumulative.values, label="SMA Crossover", linewidth=1.5)
        axes[0].plot(df["date"].values, buy_hold.values, label="Buy & Hold", linewidth=1.5, alpha=0.7)
        axes[0].set_title("Equity Curve: SMA Crossover vs Buy & Hold")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bottom: drawdown
        drawdown = cumulative / cumulative.cummax() - 1
        axes[1].fill_between(df["date"].values, drawdown.values, 0, alpha=0.4, color="red")
        axes[1].set_title("Strategy Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(os.path.dirname(__file__), "backtest_results.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"PLOT_SAVED: {plot_path}")
        print("METRICS:", json.dumps(metrics))
    """)

    # ── Step 2: Create synthetic CSV data to feed into the backtest ──
    #    In production, this would be your real Sharadar data.
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=252 * 5, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
    sample_data = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": prices})

    # ── Step 3: Run in sandbox ───────────────────────────────────────
    executor = SandboxExecutor(
        llm=None,
        max_retries=0,
        timeout=120,
        extra_packages=["pandas", "numpy", "matplotlib"],
        keep_sandbox=True,  # keep it so we can check the plot file
    )

    # We need to write the CSV into the sandbox before execution.
    # To do this, we'll use a small hook: create sandbox, write CSV,
    # then run. We can do this by manually calling the internal methods.

    # --- Manual sandbox lifecycle ---
    executor._create_sandbox()
    executor._create_venv()

    # Write the CSV into the sandbox directory
    csv_path = os.path.join(str(executor._sandbox_dir), "input_data.csv")
    sample_data.to_csv(csv_path, index=False)
    print(f"   Wrote sample data ({len(sample_data)} rows) to sandbox")

    # Install deps
    executor._install_dependencies(backtest_code)

    # Execute
    stdout, stderr, returncode = executor._execute(backtest_code)

    # ── Step 4: Check results ────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Return code: {returncode}")

    if returncode == 0:
        metrics = executor._extract_metrics(stdout)
        print(f"✅ Success!")
        print(f"   Metrics: {json.dumps(metrics, indent=2)}")

        # Check if plot was created
        plot_path = os.path.join(str(executor._sandbox_dir), "backtest_results.png")
        if os.path.exists(plot_path):
            size_kb = os.path.getsize(plot_path) / 1024
            print(f"   📊 Plot saved: {plot_path} ({size_kb:.1f} KB)")
            print(f"\n   To view the plot on your Lambda instance:")
            print(f"   scp ubuntu@<your-ip>:{plot_path} ./backtest_results.png")
        else:
            print("   ⚠️  Plot file not found")

        print(f"\n   Sandbox kept at: {executor._sandbox_dir}")
        print(f"   To clean up later: rm -rf {executor._sandbox_dir}")
    else:
        print(f"❌ Failed!")
        print(f"   Stderr: {stderr[:500]}")

    print(f"\n{'─' * 50}")

    assert returncode == 0, "Test FAILED — backtest should have succeeded"
    assert metrics is not None, "Test FAILED — no metrics parsed"
    assert os.path.exists(plot_path), "Test FAILED — plot not generated"
    print("✓ ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    print(f"\n{'SANDBOX EXECUTOR — ADVANCED TEST':^70}")
    print(f"{'Backtest with CSV data + plot output':^70}\n")
    test_backtest_with_data_and_plot()
