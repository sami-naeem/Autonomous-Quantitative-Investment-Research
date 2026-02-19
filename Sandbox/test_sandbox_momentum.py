"""
test_sandbox_momentum.py — LLM generates a backtest from a strategy description
+ smart data catalog + pandas pattern guide, runs in sandbox with retries.

Key design:
  - Smart table selection: only includes relevant tables
  - Compact schemas with explicit warnings
  - Pandas pattern guide: shows the LLM the correct structural approach for
    decile long/short backtests (pivot→signal→rank→returns) without coding
    the strategy logic itself. This prevents memory explosions and merge bugs.

Run:
    python3 test_sandbox_momentum.py --model "/models/llama-3.1-8b" --retries 5
    python3 test_sandbox_momentum.py --prompt-only
    python3 test_sandbox_momentum.py --strategy value
    python3 test_sandbox_momentum.py --strategy-text "Your custom strategy..."
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from openai import OpenAI
from sandbox_executor import SandboxExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_test")

DATA_DIR = "/home/ubuntu/data/sharadar"
DIVIDER = "=" * 70


# ─── Strategy descriptions ──────────────────────────────────────────────
STRATEGIES = {
    "momentum": textwrap.dedent("""\
        Cross-Sectional Momentum (Jegadeesh & Titman 1993)
        Universe: Active US domestic common stocks (filter via TICKERS).
        Signal: 12-1 momentum = cumulative return over months t-12 to t-1
        (skip most recent month to avoid short-term reversal).
        Portfolio: Decile sort. Long top decile, short bottom decile. Equal-weight.
        Rebalance monthly. Output: monthly long-short returns.\
    """),

    "value": textwrap.dedent("""\
        Cross-Sectional Value (Fama-French HML)
        Universe: Active US domestic common stocks with book equity data.
        Signal: Book-to-market = equity / marketcap from SF1 (dimension=ARQ).
        Portfolio: Decile sort. Long top decile (value), short bottom decile (growth).
        Equal-weight. Rebalance monthly. Output: monthly long-short returns.\
    """),

    "low_volatility": textwrap.dedent("""\
        Low-Volatility Anomaly
        Universe: Active US domestic common stocks.
        Signal: Realized volatility of daily returns over trailing 252 days.
        Portfolio: Decile sort. Long bottom decile (low vol), short top decile (high vol).
        Equal-weight. Rebalance monthly. Output: monthly long-short returns.\
    """),
}

STRATEGY_TABLES = {
    "momentum":       ["SHARADAR_TICKERS", "SHARADAR_SEP"],
    "value":          ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
    "low_volatility": ["SHARADAR_TICKERS", "SHARADAR_SEP"],
}

DEFAULT_TABLES = ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1", "SHARADAR_DAILY"]


# ─── LLM wrapper ────────────────────────────────────────────────────────
class VLLMClient:
    def __init__(self, base_url="http://localhost:8000/v1", api_key="token-abc123", model="finllama"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def invoke(self, messages):
        formatted = []
        for m in messages:
            if isinstance(m, dict):
                formatted.append(m)
            else:
                role = "user"
                cls_name = m.__class__.__name__
                if cls_name == "SystemMessage" or getattr(m, "type", "") == "system":
                    role = "system"
                elif cls_name == "HumanMessage" or getattr(m, "type", "") == "human":
                    role = "user"
                elif cls_name == "AIMessage" or getattr(m, "type", "") == "ai":
                    role = "assistant"
                formatted.append({"role": role, "content": m.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=0.2,
            max_tokens=4096,
        )

        class Resp:
            content = response.choices[0].message.content
        return Resp()


def strip_fences(text):
    text = text.strip()
    text = re.sub(r'^```(?:python)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


# ─── Scan data directory ────────────────────────────────────────────────
TABLE_PREFIXES = {
    "SHARADAR_ACTIONS": "SHARADAR_ACTIONS",
    "SHARADAR_DAILY": "SHARADAR_DAILY",
    "SHARADAR_EVENTS": "SHARADAR_EVENTS",
    "SHARADAR_INDICATORS": "SHARADAR_INDICATORS",
    "SHARADAR_METRICS": "SHARADAR_METRICS",
    "SHARADAR_SEP": "SHARADAR_SEP",
    "SHARADAR_SF1": "SHARADAR_SF1",
    "SHARADAR_SF2": "SHARADAR_SF2",
    "SHARADAR_SF3A": "SHARADAR_SF3A",
    "SHARADAR_SF3B": "SHARADAR_SF3B",
    "SHARADAR_SF3": "SHARADAR_SF3",
    "SHARADAR_SFP": "SHARADAR_SFP",
    "SHARADAR_SP500": "SHARADAR_SP500",
    "SHARADAR_TICKERS": "SHARADAR_TICKERS",
}


def scan_data_dir(data_dir):
    manifest = {}
    for f in sorted(Path(data_dir).glob("SHARADAR_*.csv")):
        for prefix, name in TABLE_PREFIXES.items():
            if f.stem.startswith(prefix):
                if name in manifest and os.path.getsize(manifest[name]) >= os.path.getsize(str(f)):
                    continue
                manifest[name] = str(f.resolve())
                break
    return manifest


def get_row_count(path):
    try:
        result = subprocess.run(
            ["wc", "-l", path], capture_output=True, text=True, timeout=30
        )
        return int(result.stdout.strip().split()[0]) - 1
    except Exception:
        return -1


# ─── Compact table descriptions ─────────────────────────────────────────
COMPACT_CATALOG = {
    "SHARADAR_SEP": (
        "Daily stock prices. Columns: ticker, date, open, high, low, close, volume, "
        "closeadj, closeunadj, lastupdated. "
        "USE closeadj for returns. LARGE — always use usecols=['ticker','date','closeadj']. "
        "Join key: ticker+date."
    ),
    "SHARADAR_TICKERS": (
        "Ticker metadata. Columns: table, permaticker, ticker, name, exchange, "
        "isdelisted, category, sector, industry, scalemarketcap. "
        "FILTER: table=='SEP' AND isdelisted=='N' AND category=='Domestic Common Stock'. "
        "Join key: ticker."
    ),
    "SHARADAR_SF1": (
        "Quarterly/annual fundamentals. Columns: ticker, dimension, calendardate, "
        "revenue, netinc, eps, pe, pb, ps, marketcap, equity, assets, debt, fcf, "
        "roe, roic, bvps, grossmargin, netmargin, de. "
        "FILTER dimension=='ARQ' for quarterly. Date column is 'calendardate' NOT 'date'. "
        "Join key: ticker."
    ),
    "SHARADAR_DAILY": (
        "Daily valuation RATIOS only. Columns: ticker, date, ev, evebit, evebitda, "
        "marketcap, pb, pe, ps. "
        "WARNING: NO price columns. Use SEP for prices. Join key: ticker+date."
    ),
    "SHARADAR_METRICS": (
        "Pre-computed technicals. Columns: ticker, date, beta1y, ma50d, ma200d, "
        "price, return1y, volume, high52w, low52w. Only ~50K rows. Join key: ticker+date."
    ),
    "SHARADAR_SP500": (
        "S&P 500 membership changes. Columns: date, action (added/removed), ticker, name."
    ),
    "SHARADAR_SFP": "Daily ETF/fund prices (same schema as SEP).",
    "SHARADAR_SF2": "Insider transactions. Columns: ticker, filingdate, transactionshares, transactionvalue.",
    "SHARADAR_SF3": "Institutional ownership detail. Columns: ticker, investorname, calendardate, value, units.",
    "SHARADAR_SF3A": "Institutional ownership by ticker. Columns: calendardate, ticker, shrholders, totalvalue.",
    "SHARADAR_SF3B": "Institutional ownership by investor. Columns: calendardate, investorname, totalvalue.",
    "SHARADAR_ACTIONS": "Corporate actions. Columns: date, action, ticker, name, value.",
    "SHARADAR_EVENTS": "Corporate events. Columns: ticker, date, eventcodes.",
    "SHARADAR_INDICATORS": "Column metadata. Rarely needed.",
}


# ─── Smart table selection ──────────────────────────────────────────────
def select_tables_for_strategy(strategy_name, strategy_text, manifest):
    if strategy_name in STRATEGY_TABLES:
        needed = STRATEGY_TABLES[strategy_name]
    else:
        text_lower = strategy_text.lower()
        needed = list(DEFAULT_TABLES)
        if any(w in text_lower for w in ["s&p 500", "sp500", "index member"]):
            needed.append("SHARADAR_SP500")
        if any(w in text_lower for w in ["insider"]):
            needed.append("SHARADAR_SF2")
        if any(w in text_lower for w in ["institutional", "ownership"]):
            needed.append("SHARADAR_SF3")
            needed.append("SHARADAR_SF3A")
        if any(w in text_lower for w in ["etf", "fund"]):
            needed.append("SHARADAR_SFP")
        if any(w in text_lower for w in ["beta", "moving average", "technical"]):
            needed.append("SHARADAR_METRICS")
    return [t for t in needed if t in manifest]


# ─── Pandas pattern guide ───────────────────────────────────────────────
PANDAS_PATTERN = textwrap.dedent("""\
    REQUIRED CODE STRUCTURE (follow this pattern exactly):

    Step 1: Load universe and prices
      tickers = pd.read_csv(TICKERS_PATH)
      tickers = tickers[(tickers['table']=='SEP') & (tickers['isdelisted']=='N') & (tickers['category']=='Domestic Common Stock')]
      valid = set(tickers['ticker'].unique())
      prices = pd.read_csv(SEP_PATH, usecols=['ticker','date','closeadj'], parse_dates=['date'])
      prices = prices[prices['ticker'].isin(valid)].sort_values(['ticker','date'])

    Step 2: Pivot to wide format (index=date, columns=tickers, values=closeadj)
      wide = prices.pivot_table(index='date', columns='ticker', values='closeadj')
      wide = wide.ffill()

    Step 3: Compute monthly prices and returns
      monthly = wide.resample('M').last()
      monthly_ret = monthly.pct_change()

    Step 4: Compute signal at each month (YOUR STRATEGY LOGIC HERE)
      Example for momentum: signal = monthly.shift(1) / monthly.shift(13) - 1
      Example for volatility: compute from daily data, then resample to monthly
      The signal should be a DataFrame with same shape as monthly (index=date, columns=tickers)

    Step 5: For each month, rank into deciles and compute long-short return
      results = []
      for t in range(len(monthly_ret)):
          sig = signal.iloc[t].dropna()
          ret = monthly_ret.iloc[t].dropna()
          common = sig.index.intersection(ret.index)
          if len(common) < 20: continue
          sig, ret = sig[common], ret[common]
          deciles = pd.qcut(sig.rank(method='first'), 10, labels=False, duplicates='drop')
          long_ret = ret[deciles == deciles.max()].mean()
          short_ret = ret[deciles == deciles.min()].mean()
          results.append({'date': monthly_ret.index[t], 'strategy_returns': long_ret - short_ret})

    Step 6: Save output
      out = pd.DataFrame(results)
      out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
      out.to_csv('returns.csv', index=False)

    CRITICAL WARNINGS:
    - NEVER merge large DataFrames on just 'ticker' without 'date' — causes cartesian product / memory explosion
    - NEVER use groupby().apply() to compute cumulative returns — use pivot + vectorized ops instead
    - The wide format (pivot_table) is the correct pattern — work with it, don't merge back
    - monthly.shift(1) gives LAST month's price — use shifts to avoid look-ahead bias\
""")


# ─── Prompt builder ─────────────────────────────────────────────────────
def build_prompt(strategy_description, manifest, selected_tables):
    file_lines = []
    for name in selected_tables:
        path = manifest[name]
        desc = COMPACT_CATALOG.get(name, "")
        n_rows = get_row_count(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        file_lines.append(
            f"  {name}\n"
            f"    Path: {path}\n"
            f"    Size: {size_mb:.0f}MB, {n_rows:,} rows\n"
            f"    {desc}"
        )

    file_listing = "\n".join(file_lines)

    prompt = textwrap.dedent(f"""\
        Write a Python backtest. Use ONLY the tables listed below.

        STRATEGY:
        {strategy_description}

        DATA FILES:
        {file_listing}

        {PANDAS_PATTERN}

        OUTPUT:
        - Save 'returns.csv' with columns: date (YYYY-MM-DD), strategy_returns (float)
        - Monthly frequency
        - Print "Backtest complete. Rows: N" and first/last 5 rows

        RULES:
        - Use ONLY pandas and numpy. All imports at top.
        - Use EXACT file paths. Use usecols= for large files.
        - Return ONLY Python code. No markdown. No explanations.
    """)

    return prompt


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Backtest generation test")
    parser.add_argument("--model", default="custom", help="vLLM model name")
    parser.add_argument("--retries", type=int, default=5, help="Max LLM fix retries")
    parser.add_argument("--timeout", type=int, default=300, help="Sandbox timeout (seconds)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--strategy", default="momentum", choices=list(STRATEGIES.keys()))
    parser.add_argument("--strategy-text", default=None, help="Custom strategy text")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    strategy_desc = args.strategy_text or STRATEGIES[args.strategy]
    strategy_label = args.strategy if not args.strategy_text else "custom"

    print(f"\n{DIVIDER}")
    print(f"BACKTEST GENERATION TEST (Real Sharadar Data)")
    print(f"Strategy:  {strategy_label}")
    print(f"Model:     {args.model}")
    print(f"Retries:   {args.retries}")
    print(f"Timeout:   {args.timeout}s")
    print(f"{DIVIDER}\n")

    # ── Scan data ───────────────────────────────────────────────────
    manifest = scan_data_dir(args.data_dir)
    print(f"Found {len(manifest)} tables on disk:")
    for name, path in sorted(manifest.items()):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name:24s} → {os.path.basename(path):55s} ({size_mb:.1f} MB)")

    if not manifest:
        print(f"\n❌ No data found in {args.data_dir}")
        sys.exit(1)

    # ── Select relevant tables ──────────────────────────────────────
    selected = select_tables_for_strategy(strategy_label, strategy_desc, manifest)
    print(f"\nSelected {len(selected)} tables for '{strategy_label}' strategy:")
    for name in selected:
        print(f"  ✓ {name}")

    # ── Build prompt ────────────────────────────────────────────────
    prompt = build_prompt(strategy_desc, manifest, selected)
    est_tokens = len(prompt) // 4 + 200
    print(f"\nPrompt: {len(prompt)} chars (~{est_tokens} tokens)")

    if est_tokens > 7000:
        print(f"⚠️  Prompt may exceed context window ({est_tokens} est. tokens)")

    if args.prompt_only:
        print(f"\n{'─' * 70}")
        print(prompt)
        print(f"{'─' * 70}")
        return

    # ── Connect to vLLM ─────────────────────────────────────────────
    print(f"\n[1] Connecting to vLLM ({args.model})...")
    llm = VLLMClient(model=args.model)
    try:
        resp = llm.invoke([{"role": "user", "content": "Say hello"}])
        print(f"    ✅ vLLM responding: {resp.content.strip()[:50]}")
    except Exception as e:
        print(f"    ❌ Cannot reach vLLM: {e}")
        sys.exit(1)

    # ── Generate code ───────────────────────────────────────────────
    print(f"\n[2] Prompting LLM to generate backtest code...")

    system_msg = (
        "You are an expert quantitative finance Python developer. "
        "Write clean backtest code using pandas and numpy. "
        "Follow the REQUIRED CODE STRUCTURE in the prompt exactly. "
        "Return ONLY executable Python code. No markdown fences. No explanations."
    )

    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ])

    code = strip_fences(resp.content.strip())

    print(f"\n[3] Generated {len(code)} chars ({len(code.splitlines())} lines)")
    print(f"\n{'─' * 70}")
    print("GENERATED CODE (first 60 lines):")
    print(f"{'─' * 70}")
    for i, line in enumerate(code.split("\n")[:60], 1):
        print(f"  {i:3d} | {line}")
    total_lines = len(code.splitlines())
    if total_lines > 60:
        print(f"  ... ({total_lines} total lines)")
    print(f"{'─' * 70}")

    with open("generated_backtest.py", "w") as f:
        f.write(code)
    print(f"💾 Saved to: generated_backtest.py\n")

    # ── Execute in sandbox ──────────────────────────────────────────
    print(f"[4] Running in sandbox (max_retries={args.retries}, timeout={args.timeout}s)...")
    print(f"    LLM diagnoses and fixes errors on failure.\n")

    executor = SandboxExecutor(
        llm=llm,
        max_retries=args.retries,
        timeout=args.timeout,
        extra_packages=["pandas", "numpy"],
        keep_sandbox=True,
    )

    result = executor.run(code)

    # ── Report ──────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"RESULTS")
    print(f"{DIVIDER}")
    print(f"  Success:       {result['success']}")
    print(f"  Retries used:  {result['retries_used']} / {args.retries}")

    if result["success"]:
        stdout = result["stdout"]
        print(f"\n  ─── STDOUT ({len(stdout)} chars) ───")
        if len(stdout) > 2000:
            print(stdout[:1000])
            print(f"\n  ... (truncated) ...\n")
            print(stdout[-800:])
        else:
            print(stdout)
        print(f"  ─── END STDOUT ───")

        if result.get("returns_csv"):
            print(f"\n  📊 Returns CSV: {result['returns_csv']}")
        if result.get("output_files"):
            print(f"  📁 Output files: {result['output_files']}")

        _validate_returns_csv(result)

        if result["retries_used"] > 0:
            with open("generated_backtest_fixed.py", "w") as f:
                f.write(result["code_final"])
            print(f"\n  💾 Fixed code saved to: generated_backtest_fixed.py")
    else:
        print(f"\n  ❌ FAILED after {result['retries_used']} retries")
        print(f"\n  Last error:")
        print(textwrap.indent(result['stderr'][:1000], '    '))

    if result["error_history"]:
        print(f"\n  ─── Error History ({len(result['error_history'])} attempts) ───")
        for i, err in enumerate(result["error_history"], 1):
            err_lines = err.strip().split("\n")
            last_line = err_lines[-1] if err_lines else err[:100]
            print(f"    Attempt {i}: {last_line[:150]}")
        print(f"  ─── End Error History ───")

    if result.get("sandbox_path"):
        print(f"\n  Sandbox: {result['sandbox_path']}")
        print(f"  Clean up: rm -rf {result['sandbox_path']}")

    print(f"\n{DIVIDER}")
    if result["success"]:
        print(f"🎉 SUCCESS — {strategy_label} backtest completed!")
    else:
        print(f"💥 FAILED — Check generated_backtest.py and errors above")
    print(f"{DIVIDER}\n")

    return result


# ─── Output validation ──────────────────────────────────────────────────
def _validate_returns_csv(result):
    import pandas as pd

    csv_path = result.get("returns_csv")
    if not csv_path or not os.path.exists(csv_path):
        sandbox = result.get("sandbox_path", "")
        if sandbox:
            candidate = os.path.join(sandbox, "returns.csv")
            if os.path.exists(candidate):
                csv_path = candidate

    if not csv_path or not os.path.exists(csv_path):
        print(f"\n  ⚠️  returns.csv not found")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"\n  ─── RETURNS CSV VALIDATION ───")
        print(f"  Shape: {df.shape}  Columns: {list(df.columns)}")

        has_date = 'date' in df.columns
        has_returns = 'strategy_returns' in df.columns
        print(f"  date col: {'✅' if has_date else '❌'}  strategy_returns col: {'✅' if has_returns else '❌'}")

        if has_date and has_returns and len(df) > 0:
            dates = pd.to_datetime(df['date'])
            rets = df['strategy_returns'].astype(float)

            print(f"  Range: {dates.min():%Y-%m-%d} → {dates.max():%Y-%m-%d}  ({len(df)} months)")
            print(f"  Mean: {rets.mean():.5f}  Std: {rets.std():.5f}  Min: {rets.min():.5f}  Max: {rets.max():.5f}")

            cum = (1 + rets).cumprod()
            sharpe = rets.mean() / rets.std() * (12 ** 0.5) if rets.std() > 0 else 0
            total_ret = cum.iloc[-1] - 1
            max_dd = (cum / cum.cummax() - 1).min()
            print(f"  Sharpe: {sharpe:.3f}  Total: {total_ret:.3f}  MaxDD: {max_dd:.3f}")

            issues = []
            if abs(rets.mean()) > 0.20:
                issues.append(f"extreme mean ({rets.mean():.4f})")
            if len(df) < 12:
                issues.append(f"few months ({len(df)})")
            if rets.std() == 0:
                issues.append("zero variance")
            if rets.isna().any():
                issues.append(f"{rets.isna().sum()} NaN")

            for issue in issues:
                print(f"  ⚠️  {issue}")
            if not issues:
                print(f"  ✅ Returns look reasonable")

        elif len(df) == 0:
            print(f"  ❌ Empty DataFrame")

        print(f"  ─── END VALIDATION ───")
    except Exception as e:
        print(f"\n  ⚠️  Validation error: {e}")


if __name__ == "__main__":
    main()