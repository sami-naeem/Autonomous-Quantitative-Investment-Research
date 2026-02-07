"""
test_sandbox.py — Standalone test for SandboxExecutor on your Lambda instance.

No LangGraph, no GCS, no project dependencies needed.
Just drop this file next to sandbox_executor.py and run:

    python test_sandbox.py

It runs 3 tests:
  1. Happy path  — clean backtest code that should succeed on attempt 1
  2. Broken code — deliberate bug, no LLM attached → shows graceful failure
  3. Broken code + mock LLM fixer — simulates the LLM fixing the bug
"""

import json
import logging
import textwrap

from sandbox_executor import SandboxExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

DIVIDER = "=" * 70


# ─── Test 1: Happy path ─────────────────────────────────────────────────
def test_happy_path():
    print(f"\n{DIVIDER}")
    print("TEST 1 — Happy path (clean backtest code)")
    print(DIVIDER)

    code = textwrap.dedent("""\
        import json
        import numpy as np
        import pandas as pd

        # --- Simulate a simple momentum backtest ---
        np.random.seed(42)
        n_days = 252 * 3  # 3 years of trading days

        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days)))

        df = pd.DataFrame({"date": dates, "price": prices})
        df["returns"] = df["price"].pct_change()
        df["momentum_signal"] = df["price"].pct_change(21)  # 1-month momentum
        df["position"] = np.where(df["momentum_signal"] > 0, 1, -1)
        df["strategy_returns"] = df["position"].shift(1) * df["returns"]
        df = df.dropna()

        # Compute metrics
        ann_return = df["strategy_returns"].mean() * 252
        ann_vol = df["strategy_returns"].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        cumulative = (1 + df["strategy_returns"]).cumprod()
        max_dd = float((cumulative / cumulative.cummax() - 1).min())
        total_return = float(cumulative.iloc[-1] - 1)

        metrics = {
            "sharpe_ratio": round(sharpe, 4),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(total_return, 4),
            "n_trading_days": len(df),
        }
        print("METRICS:", json.dumps(metrics))
    """)

    executor = SandboxExecutor(llm=None, max_retries=0, timeout=120)
    result = executor.run(code)

    print(f"\n✅ Success: {result['success']}")
    print(f"   Metrics: {json.dumps(result['metrics'], indent=2)}")
    assert result["success"], "Test 1 FAILED — expected success"
    assert result["metrics"] is not None, "Test 1 FAILED — no metrics parsed"
    print("   ✓ PASSED")


# ─── Test 2: Broken code, no LLM ────────────────────────────────────────
def test_broken_no_llm():
    print(f"\n{DIVIDER}")
    print("TEST 2 — Broken code, no LLM (should fail gracefully)")
    print(DIVIDER)

    broken_code = textwrap.dedent("""\
        import json
        import pandas as pd

        # Bug: 'prices' is never defined
        df = pd.DataFrame({"price": prices})
        sharpe = df["price"].mean()
        print("METRICS:", json.dumps({"sharpe": sharpe}))
    """)

    executor = SandboxExecutor(llm=None, max_retries=2, timeout=30)
    result = executor.run(broken_code)

    print(f"\n❌ Success: {result['success']}  (expected False)")
    print(f"   Retries used: {result['retries_used']}")
    print(f"   Errors captured: {len(result['error_history'])}")
    print(f"   Last error: {result['stderr'][:200]}")
    assert not result["success"], "Test 2 FAILED — should not have succeeded"
    assert len(result["error_history"]) > 0, "Test 2 FAILED — no errors captured"
    print("   ✓ PASSED")


# ─── Test 3: Broken code + mock LLM that fixes it ───────────────────────
class MockFixerLLM:
    """
    Simulates an LLM that receives broken code and returns a fixed version.
    Drop-in replacement for a LangChain LLM — just needs .invoke().
    
    On your Lambda instance, replace this with your real LLM:
        from Agents.llm_config import get_llm_by_name
        llm = get_llm_by_name("openai")  # or your fine-tune
    """

    def invoke(self, messages):
        # The system+human messages come in as a list; we just return fixed code
        fixed = textwrap.dedent("""\
            import json
            import numpy as np
            import pandas as pd

            # Fixed: define prices properly
            np.random.seed(99)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 500)))
            df = pd.DataFrame({"price": prices})
            df["returns"] = df["price"].pct_change().dropna()
            sharpe = float(df["returns"].mean() / df["returns"].std() * np.sqrt(252))
            total_return = float(df["price"].iloc[-1] / df["price"].iloc[0] - 1)

            metrics = {
                "sharpe_ratio": round(sharpe, 4),
                "total_return": round(total_return, 4),
            }
            print("METRICS:", json.dumps(metrics))
        """)

        class FakeResponse:
            content = fixed

        return FakeResponse()


def test_broken_with_llm_fix():
    print(f"\n{DIVIDER}")
    print("TEST 3 — Broken code + mock LLM fixer (should succeed on retry)")
    print(DIVIDER)

    broken_code = textwrap.dedent("""\
        import json
        import pandas as pd

        # Bug: 'prices' is never defined
        df = pd.DataFrame({"price": prices})
        sharpe = df["price"].mean()
        print("METRICS:", json.dumps({"sharpe": sharpe}))
    """)

    mock_llm = MockFixerLLM()
    executor = SandboxExecutor(llm=mock_llm, max_retries=3, timeout=60)
    result = executor.run(broken_code)

    print(f"\n✅ Success: {result['success']}")
    print(f"   Retries used: {result['retries_used']}")
    print(f"   Metrics: {json.dumps(result['metrics'], indent=2)}")
    assert result["success"], "Test 3 FAILED — LLM fix should have worked"
    assert result["retries_used"] >= 1, "Test 3 FAILED — should have needed a retry"
    print("   ✓ PASSED")


# ─── Run all ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'SANDBOX EXECUTOR — STANDALONE TEST SUITE':^70}")
    print(f"{'Drop this next to sandbox_executor.py and run: python test_sandbox.py':^70}\n")

    test_happy_path()
    test_broken_no_llm()
    test_broken_with_llm_fix()

    print(f"\n{DIVIDER}")
    print("ALL 3 TESTS PASSED ✓")
    print(DIVIDER)
