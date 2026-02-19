"""
test_sandbox_sharadar.py — LLM picks Sharadar tables and writes load/merge code.

Flow:
  1. Read Data_Dictionary_Sharadar.md for schema
  2. Scan /home/ubuntu/data/sharadar for CSVs → file paths
  3. Give LLM a task → it picks tables, writes code to load/merge/print head
  4. Sandbox runs it

Run:
    python3 test_sandbox_sharadar.py
    python3 test_sandbox_sharadar.py --task sp500_prices
    python3 test_sandbox_sharadar.py --model finllama
"""

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from pathlib import Path

from openai import OpenAI
from sandbox_executor import SandboxExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

DATA_DIR = "/home/ubuntu/data/sharadar"
DATA_DICT_PATH = "/home/ubuntu/capstone/Data_Dictionary_Sharadar.md"

# Compact schema: just the essentials for the LLM to pick tables and merge correctly
COMPACT_SCHEMA = {
    "SHARADAR_SEP":     "Daily equity prices. Key cols: ticker, date, open, high, low, close, volume, closeadj, closeunadj",
    "SHARADAR_SFP":     "Daily fund/ETF prices. Key cols: ticker, date, open, high, low, close, volume, closeadj",
    "SHARADAR_SF1":     "Quarterly/annual fundamentals. Key cols: ticker, dimension(ARQ=quarterly,ARY=annual), calendardate, revenue, netinc, ebitda, eps, pe, pb, ps, marketcap, roe, roic, grossmargin, netmargin, de, debt, equity, assets, fcf",
    "SHARADAR_SF2":     "Insider transactions (SEC forms 3/4/5). Key cols: ticker, filingdate, ownername, officertitle, transactionshares, transactionvalue, transactionpricepershare",
    "SHARADAR_SF3":     "Institutional ownership detail. Key cols: ticker, investorname, securitytype, calendardate, value, units",
    "SHARADAR_SF3A":    "Institutional ownership by ticker. Key cols: calendardate, ticker, shrholders, shrvalue, totalvalue",
    "SHARADAR_SF3B":    "Institutional ownership by investor. Key cols: calendardate, investorname, shrholdings, totalvalue",
    "SHARADAR_DAILY":   "Daily valuation ratios. Key cols: ticker, date, ev, evebit, evebitda, marketcap, pb, pe, ps",
    "SHARADAR_TICKERS": "Ticker metadata. Key cols: ticker, name, exchange, isdelisted(Y/N), category, sector, industry, scalemarketcap, location",
    "SHARADAR_SP500":   "S&P 500 membership. Key cols: date, action(added/removed), ticker, name",
    "SHARADAR_ACTIONS": "Corporate actions. Key cols: date, action, ticker, name, value",
    "SHARADAR_METRICS": "Market technicals. Key cols: ticker, date, ma50d, ma200d, price, return1y, volume, beta1y, high52w, low52w",
    "SHARADAR_EVENTS":  "Corporate events. Key cols: ticker, date, eventcodes",
    "SHARADAR_INDICATORS": "Metadata table describing indicators (rarely needed).",
}

DIVIDER = "=" * 70


# ─── LLM wrapper (same as test_retries.py) ──────────────────────────────
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


# ─── Scan data directory ─────────────────────────────────────────────────
TABLE_PREFIXES = {
    "SHARADAR_ACTIONS": "SHARADAR_ACTIONS", "SHARADAR_DAILY": "SHARADAR_DAILY",
    "SHARADAR_EVENTS": "SHARADAR_EVENTS",   "SHARADAR_INDICATORS": "SHARADAR_INDICATORS",
    "SHARADAR_METRICS": "SHARADAR_METRICS", "SHARADAR_SEP": "SHARADAR_SEP",
    "SHARADAR_SF1": "SHARADAR_SF1",         "SHARADAR_SF2": "SHARADAR_SF2",
    "SHARADAR_SF3A": "SHARADAR_SF3A",       "SHARADAR_SF3B": "SHARADAR_SF3B",
    "SHARADAR_SF3": "SHARADAR_SF3",         "SHARADAR_SFP": "SHARADAR_SFP",
    "SHARADAR_SP500": "SHARADAR_SP500",     "SHARADAR_TICKERS": "SHARADAR_TICKERS",
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


# ─── Tasks ───────────────────────────────────────────────────────────────
TASKS = {
    "prices_and_fundamentals": (
        "I need daily stock prices merged with quarterly fundamentals (like revenue, "
        "net income, PE ratio) for each ticker. Load the right tables, merge them "
        "appropriately, and print the first 10 rows."
    ),
    "insider_activity": (
        "I want to see insider trading activity alongside company metadata (sector, "
        "industry, market cap scale). Load what you need, merge, and print the head."
    ),
    "institutional_ownership": (
        "I want to see which institutional investors hold which stocks, along with "
        "the company name and sector. Load the right tables, merge, print the head."
    ),
    "sp500_prices": (
        "I need daily prices only for stocks that are currently in the S&P 500. "
        "Load what you need, filter, merge, and print the first 10 rows."
    ),
    "valuation_screen": (
        "I want to screen stocks by valuation — I need PE, PB, PS ratios alongside "
        "the company sector and industry. Load the right tables, merge, show the head."
    ),
}


# ─── Prompt ──────────────────────────────────────────────────────────────
def build_prompt(task_description, manifest):
    file_listing = "\n".join(
        f"  {name}: {path}\n    → {COMPACT_SCHEMA.get(name, 'Unknown')}"
        for name, path in sorted(manifest.items())
    )

    return textwrap.dedent(f"""\
        TASK: {task_description}

        AVAILABLE CSV FILES:
        {file_listing}

        Write a Python script that:
        1. Prints which tables you chose and why (one line each)
        2. Loads only the needed CSVs using pd.read_csv(path, nrows=10000)
        3. If multiple tables, merge on appropriate keys (usually ticker, or ticker+date)
        4. Prints the .shape and .head(10) of the final dataframe

        Use only pandas. Return ONLY Python code.
    """)


# ─── Run a single task ──────────────────────────────────────────────────
def run_task(task_name, task_desc, manifest, llm):
    print(f"\n{DIVIDER}")
    print(f"TASK: {task_name}")
    print(f"  {task_desc[:80]}...")
    print(f"{DIVIDER}")

    prompt = build_prompt(task_desc, manifest)

    print(f"\n[1] Prompting LLM to select tables and write code...")
    resp = llm.invoke([
        {"role": "system", "content": "You are a Python code generator. Return ONLY valid Python code. No markdown fences, no explanations."},
        {"role": "user", "content": prompt},
    ])
    code = strip_fences(resp.content.strip())

    print(f"[2] LLM generated {len(code)} chars")
    print(f"\n--- LLM CODE ---")
    print(code)
    print(f"--- END CODE ---\n")

    print(f"[3] Running in sandbox...")
    executor = SandboxExecutor(
        llm=llm,
        max_retries=1,
        timeout=120,
        extra_packages=["pandas"],
        keep_sandbox=False,
    )
    result = executor.run(code)

    print(f"\n{'─' * 50}")
    if result["success"]:
        print(f"✅ SUCCESS (retries: {result['retries_used']})")
        print(f"\n--- OUTPUT ---")
        print(result["stdout"])
        print(f"--- END ---")
    else:
        print(f"❌ FAILED (retries: {result['retries_used']})")
        print(f"Error: {result['stderr'][:500]}")
    print(f"{'─' * 50}")

    return result["success"]


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS.keys()) + ["all"], default="all")
    parser.add_argument("--model", default="custom", help="vLLM model name")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    print(f"\n{DIVIDER}")
    print(f"SHARADAR DATA SELECTION TEST")
    print(f"Model: {args.model}")
    print(f"{DIVIDER}\n")

    # Scan data
    manifest = scan_data_dir(args.data_dir)
    print(f"Found {len(manifest)} tables:")
    for name, path in sorted(manifest.items()):
        print(f"  {name:24s} → {os.path.basename(path)}")

    if not manifest:
        print(f"No data found in {args.data_dir}")
        sys.exit(1)

    # Connect to vLLM
    print(f"\nConnecting to vLLM ({args.model})...")
    llm = VLLMClient(model=args.model)
    try:
        resp = llm.invoke([{"role": "user", "content": "Say hello"}])
        print(f"  ✅ vLLM responding: {resp.content.strip()[:50]}\n")
    except Exception as e:
        print(f"  ❌ Cannot reach vLLM: {e}")
        sys.exit(1)

    # Run tasks
    tasks = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    results = {}
    for name, desc in tasks.items():
        results[name] = run_task(name, desc, manifest, llm)

    # Summary
    print(f"\n{DIVIDER}")
    print(f"SUMMARY")
    print(f"{DIVIDER}")
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'}  {name}")
    print(f"\n  {sum(results.values())}/{len(results)} passed")
    print(f"{DIVIDER}")


if __name__ == "__main__":
    main()
