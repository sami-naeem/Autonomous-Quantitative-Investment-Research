"""
run_replication.py — Downstream replication-code generator & sandbox executor.

Reads the upstream Agent 1 (theme_pick) + Agent 2 (factor_extract) JSON outputs
for each paper from GCS, uses those to build a strategy description + data
catalog prompt, calls the LLM to generate backtest code against real Sharadar
data, runs the code inside SandboxExecutor with LLM-assisted retry logic, and
writes all artifacts back to GCS and/or locally.

GCS layout consumed (upstream):
    gs://<OUTPUT_BUCKET>/<paper_name>/<model_choice>/<timestamp>.json
    Example: investment-research-output/ref_Abarbanell_and_Bushee_1998/finllama/2025-06-15T14-22-09.json

GCS layout produced (downstream):
    gs://<OUTPUT_BUCKET>/replication_outputs/<paper_name>/<timestamp>.json       (artifact)
    gs://<OUTPUT_BUCKET>/replication_outputs/<paper_name>/<timestamp>_code.py    (standalone code)
    gs://<OUTPUT_BUCKET>/replication_outputs/<paper_name>/<timestamp>_returns.csv (execution output)

Usage:
    python run_replication.py --paper ref_Abarbanell_and_Bushee_1998
    python run_replication.py --paper ref_Abarbanell_and_Bushee_1998 --model finllama
    python run_replication.py --all
    python run_replication.py --all --start-index 5
    python run_replication.py --paper ref_Abarbanell_and_Bushee_1998 --dry-run
    python run_replication.py --paper ref_Abarbanell_and_Bushee_1998 --prompt-only
    python run_replication.py --list-papers
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from google.cloud import storage

# ── Project imports (graceful fallback) ──────────────────────────────────
try:
    from Agents.config import GCSConfig, LLMConfig
except ImportError:
    GCSConfig = None  # type: ignore[assignment,misc]
    LLMConfig = None  # type: ignore[assignment,misc]

try:
    from Agents.llm_config import get_llm_by_name
except ImportError:
    get_llm_by_name = None  # type: ignore[assignment,misc]

try:
    from Agents.gcs_utils import get_storage_client
except ImportError:
    get_storage_client = None  # type: ignore[assignment,misc]


# ═════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_replication")


# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════
if GCSConfig is not None:
    GCS_PROJECT_ID = GCSConfig.PROJECT_ID
    GCS_CREDENTIALS_PATH = GCSConfig.CREDENTIALS_PATH or getattr(
        GCSConfig, "LAMBDA_CREDENTIALS_PATH", None
    )
    OUTPUT_BUCKET = getattr(
        GCSConfig, "OUTPUT_BUCKET_NAME", "investment-research-output"
    )
else:
    GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "")
    GCS_CREDENTIALS_PATH = os.getenv("GCS_CREDENTIALS_PATH", "")
    OUTPUT_BUCKET = os.getenv("GCS_OUTPUT_BUCKET", "investment-research-output")

# Upstream outputs sit at bucket root: <paper_name>/<model>/<ts>.json
UPSTREAM_PREFIX = ""

# Replication outputs go under a dedicated prefix
REPLICATION_PREFIX = os.getenv("GCS_REPLICATION_PREFIX", "replication_outputs/")

# Sharadar data directory (on Lambda / local dev)
DATA_DIR = os.getenv("SHARADAR_DATA_DIR", "/home/ubuntu/data/sharadar")

# Defaults
DEFAULT_MODEL = "finllama"
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 300
OUTER_RETRIES = int(os.getenv("REPLICATION_OUTER_RETRIES", "1"))


# ═════════════════════════════════════════════════════════════════════════
# GCS client
# ═════════════════════════════════════════════════════════════════════════
def _get_gcs_client() -> Optional[storage.Client]:
    """Return an authenticated GCS client, or None on failure."""
    if get_storage_client is not None:
        client = get_storage_client()
        if client is not None:
            return client
    try:
        creds = GCS_CREDENTIALS_PATH
        if creds and Path(creds).exists():
            return storage.Client.from_service_account_json(creds)
        return storage.Client(project=GCS_PROJECT_ID)
    except Exception as e:
        logger.warning(f"GCS client creation failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════
# 1. load_inputs — fetch upstream Agent 1+2 JSONs
# ═════════════════════════════════════════════════════════════════════════
def _find_latest_upstream_blob(
    client: storage.Client,
    paper_name: str,
    model_choice: Optional[str] = None,
) -> Optional[storage.Blob]:
    """Find the most-recent agent output JSON for *paper_name* in GCS.

    Upstream save_output_to_gcs_node writes blobs at:
        <paper_name>/<model_choice>/<timestamp>.json
    We pick the most recently *created* blob (by GCS time_created).
    """
    bucket = client.bucket(OUTPUT_BUCKET)
    if model_choice:
        prefix = f"{UPSTREAM_PREFIX}{paper_name}/{model_choice}/"
    else:
        prefix = f"{UPSTREAM_PREFIX}{paper_name}/"

    blobs = list(bucket.list_blobs(prefix=prefix))
    json_blobs = [b for b in blobs if b.name.endswith(".json")]
    if not json_blobs:
        return None
    json_blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return json_blobs[0]


def load_inputs(
    paper_name: str,
    model_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """Load upstream agent outputs (theme_pick + factor_extract) from GCS.

    Downloads the most recently created JSON blob at:
        gs://<OUTPUT_BUCKET>/<paper_name>/<model_choice>/<timestamp>.json

    Returns dict with: paper_name, model_choice, theme_pick, factor_extract
    """
    client = _get_gcs_client()
    if client is None:
        raise ConnectionError(
            f"Cannot load inputs for '{paper_name}': GCS client unavailable."
        )

    blob = _find_latest_upstream_blob(client, paper_name, model_choice)
    if blob is None:
        raise FileNotFoundError(
            f"No upstream output in gs://{OUTPUT_BUCKET}/{paper_name}/"
        )

    logger.info(f"[load_inputs] Downloading gs://{OUTPUT_BUCKET}/{blob.name}")
    raw = json.loads(blob.download_as_text())
    return _normalise_upstream(raw)


def _normalise_upstream(raw: dict) -> dict:
    """Ensure theme_pick/factor_extract are dicts (batch runner serialises
    them as JSON strings)."""
    for key in ("theme_pick", "factor_extract"):
        val = raw.get(key, {})
        if isinstance(val, str):
            try:
                raw[key] = json.loads(val)
            except json.JSONDecodeError:
                raw[key] = {"_raw": val}
    return raw


# ═════════════════════════════════════════════════════════════════════════
# Sharadar Data Catalog  (mirrors test_sandbox_momentum.py)
# ═════════════════════════════════════════════════════════════════════════
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
    "SHARADAR_SF2": (
        "Insider transactions. Columns: ticker, filingdate, transactionshares, transactionvalue."
    ),
    "SHARADAR_SF3": (
        "Institutional ownership detail. Columns: ticker, investorname, calendardate, value, units."
    ),
    "SHARADAR_SF3A": (
        "Institutional ownership by ticker. Columns: calendardate, ticker, shrholders, totalvalue."
    ),
    "SHARADAR_SF3B": (
        "Institutional ownership by investor. Columns: calendardate, investorname, totalvalue."
    ),
    "SHARADAR_ACTIONS": "Corporate actions. Columns: date, action, ticker, name, value.",
    "SHARADAR_EVENTS": "Corporate events. Columns: ticker, date, eventcodes.",
    "SHARADAR_INDICATORS": "Column metadata. Rarely needed.",
}

# Theme → tables mapping for common themes.
# Papers whose theme matches a key here get those tables automatically.
# All others get the LLM-driven selection as fallback.
THEME_TABLE_HINTS: Dict[str, List[str]] = {
    "Momentum":       ["SHARADAR_TICKERS", "SHARADAR_SEP"],
    "Seasonality":    ["SHARADAR_TICKERS", "SHARADAR_SEP"],
    "Size":           ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_DAILY"],
    "Value":          ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1", "SHARADAR_DAILY"],
    "Profitability":  ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
    "Quality":        ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
    "Investment":     ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
    "Accruals":       ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
    "Low Leverage":   ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
    "Debt Issuance":  ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1"],
}

# Fallback: if theme not in the map above, give the LLM these core tables
# plus any extras it needs (SEP + TICKERS are always included).
DEFAULT_TABLES = ["SHARADAR_TICKERS", "SHARADAR_SEP", "SHARADAR_SF1", "SHARADAR_DAILY"]


# ── Scan local Sharadar data directory ───────────────────────────────────
def scan_data_dir(data_dir: str) -> Dict[str, str]:
    """Return {TABLE_NAME: absolute_path} for every Sharadar CSV found."""
    manifest: Dict[str, str] = {}
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return manifest

    for f in sorted(data_path.glob("SHARADAR_*.csv")):
        for prefix, name in TABLE_PREFIXES.items():
            if f.stem.startswith(prefix):
                # Keep the largest file if there are duplicates
                if name in manifest and os.path.getsize(manifest[name]) >= os.path.getsize(str(f)):
                    continue
                manifest[name] = str(f.resolve())
                break
    return manifest


def get_row_count(path: str) -> int:
    """Fast row count via wc -l."""
    try:
        result = subprocess.run(
            ["wc", "-l", path], capture_output=True, text=True, timeout=30
        )
        return int(result.stdout.strip().split()[0]) - 1
    except Exception:
        return -1


# ── Table selection driven by upstream JSONs ─────────────────────────────
def select_tables(
    upstream: Dict[str, Any],
    manifest: Dict[str, str],
) -> List[str]:
    """Choose which Sharadar tables the backtest needs.

    Strategy:
        1. If the theme_pick.theme matches a known theme → use the curated list.
        2. Otherwise, use DEFAULT_TABLES as baseline and extend based on signals
           in factor_extract.computation_logic / purpose text.
        3. Always filter to tables actually present on disk (manifest).
    """
    theme = upstream.get("theme_pick", {}).get("theme", "")
    fe = upstream.get("factor_extract", {})
    purpose = fe.get("purpose", "")
    logic = fe.get("computation_logic", "")
    combined_text = f"{purpose} {logic} {theme}".lower()

    # 1. Try curated theme map
    if theme in THEME_TABLE_HINTS:
        needed = list(THEME_TABLE_HINTS[theme])
    else:
        needed = list(DEFAULT_TABLES)

    # 2. Extend based on content in factor_extract
    if any(w in combined_text for w in ["s&p 500", "sp500", "index member"]):
        needed.append("SHARADAR_SP500")
    if any(w in combined_text for w in ["insider", "insider trading"]):
        needed.append("SHARADAR_SF2")
    if any(w in combined_text for w in ["institutional", "ownership", "13f"]):
        needed.append("SHARADAR_SF3")
        needed.append("SHARADAR_SF3A")
    if any(w in combined_text for w in ["etf", "fund price"]):
        needed.append("SHARADAR_SFP")
    if any(w in combined_text for w in ["beta", "moving average", "technical indicator"]):
        needed.append("SHARADAR_METRICS")
    if any(w in combined_text for w in [
        "earnings", "revenue", "book", "equity", "debt", "assets",
        "profitab", "margin", "accrual", "fundamental", "leverage",
        "roe", "roic", "fcf", "issuance",
    ]):
        if "SHARADAR_SF1" not in needed:
            needed.append("SHARADAR_SF1")
    if any(w in combined_text for w in [
        "valuation", "pe ratio", "pb ratio", "enterprise value", "ev/ebit",
        "market cap", "marketcap",
    ]):
        if "SHARADAR_DAILY" not in needed:
            needed.append("SHARADAR_DAILY")

    # 3. TICKERS + SEP are always required
    for must_have in ["SHARADAR_TICKERS", "SHARADAR_SEP"]:
        if must_have not in needed:
            needed.insert(0, must_have)

    # Deduplicate, preserve order, filter to disk
    seen: Set[str] = set()
    final: List[str] = []
    for t in needed:
        if t not in seen and t in manifest:
            final.append(t)
            seen.add(t)

    return final


# ═════════════════════════════════════════════════════════════════════════
# 2. build_prompt — strategy description + data catalog + pandas pattern
# ═════════════════════════════════════════════════════════════════════════
PANDAS_PATTERN = textwrap.dedent("""\
 REQUIRED CODE STRUCTURE (follow this pattern exactly):

Step 1: Load universe and prices
  tickers = pd.read_csv(TICKERS_PATH)
  tickers = tickers[(tickers['table']=='SEP') & (tickers['isdelisted']=='N') & (tickers['category']=='Domestic Common Stock')]
  valid = set(tickers['ticker'].unique())
  prices = pd.read_csv(SEP_PATH, usecols=['ticker','date','closeadj'], parse_dates=['date'])
  prices = prices[prices['ticker'].isin(valid)].sort_values(['ticker','date'])

Step 2: Pivot to wide format (index=date, columns=tickers, values=closeadj)
  wide = prices.pivot(index='date', columns='ticker', values='closeadj')
  wide = wide.sort_index()
  wide = wide.ffill()

Step 3: Compute monthly prices and returns (USE 'ME' ONLY)
  monthly = wide.resample('ME').last()
  monthly_ret = monthly.pct_change()

Step 4: Compute signal at each month (YOUR STRATEGY LOGIC HERE)
  The signal MUST be a DataFrame with the same shape/labels as monthly:
    signal = pd.DataFrame(index=monthly.index, columns=monthly.columns, dtype='float64')

  If the paper uses SF1 fundamentals, you MUST align quarterly data onto the monthly grid like this:

    sf1 = pd.read_csv(SF1_PATH, parse_dates=['calendardate'])
    sf1 = sf1[sf1['dimension']=='ARQ']
    sf1 = sf1[sf1['ticker'].isin(valid)].sort_values(['ticker','calendardate'])

    month_df = monthly.index.to_frame(index=False, name='date')

    for ticker in monthly.columns:
        td = sf1[sf1['ticker']==ticker][['calendardate', <FIELDS_YOU_NEED>]].dropna(subset=['calendardate'])
        if td.empty:
            continue
        td = td.sort_values('calendardate')

        m = pd.merge_asof(
            month_df,
            td,
            left_on='date',
            right_on='calendardate',
            direction='backward'
        )

        # CRITICAL: assign positionally to avoid index-alignment NaNs
        signal[ticker] = m[<SIGNAL_COLUMN>].to_numpy()

  After building signal you MUST compute and STORE (do NOT print) diagnostics:
    signal_coverage = float(signal.notna().mean().mean())

Step 5: For each month, rank into deciles and compute long-short return
  results = []
  skipped_common_lt20 = 0
  common_sizes = []

  for t in range(len(monthly_ret)):
      sig = signal.iloc[t].dropna()
      ret = monthly_ret.iloc[t].dropna()
      common = sig.index.intersection(ret.index)

      common_sizes.append(len(common))
      if len(common) < 20:
          skipped_common_lt20 += 1
          continue

      sig, ret = sig[common], ret[common]
      deciles = pd.qcut(sig.rank(method='first'), 10, labels=False, duplicates='drop')
      long_ret = ret[deciles == deciles.max()].mean()
      short_ret = ret[deciles == deciles.min()].mean()
      results.append({'date': monthly_ret.index[t], 'strategy_returns': long_ret - short_ret})

  avg_common = float(np.mean(common_sizes)) if common_sizes else 0.0

Step 6: Save output (ALWAYS CREATE returns.csv)
  out = pd.DataFrame(results)

  if not out.empty:
      out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
  else:
      out = pd.DataFrame(columns=['date','strategy_returns'])

  out.to_csv('returns.csv', index=False)

Step 7: Metrics (MUST store as a single JSON object printed on one line)
  Build a dict called metrics_dict that ALWAYS includes these diagnostics:
    - 'signal_coverage': signal_coverage
    - 'skipped_common_lt20': skipped_common_lt20
    - 'avg_common': avg_common
    - 'rows': int(len(out))

  If len(out) > 0, ALSO include:
    - 'sharpe_ratio'
    - 'annualized_return'
    - 'total_return'
    - 'max_drawdown'
    - 'n_months'

  If len(out) == 0, set the performance metrics to None.

  Finally, print EXACTLY ONE LINE in this format (no other prints):
    print("METRICS:", json.dumps(metrics_dict))

CRITICAL WARNINGS:
- NEVER merge large DataFrames on just 'ticker' without 'date' — causes cartesian product / memory explosion
- NEVER use groupby().apply() to compute cumulative returns — use pivot + vectorized ops instead
- The wide format (pivot) is the correct pattern — work with it, don't merge back to long unless necessary
- Use shifts/lags to avoid look-ahead bias; for fundamentals ALWAYS use the MOST RECENT value AVAILABLE BEFORE the signal date
- For pd.merge_asof you MUST:
    (1) sort by the date key
    (2) use direction='backward'
    (3) assign with .to_numpy() to avoid index-alignment NaNs\
""")


def _build_strategy_description(upstream: Dict[str, Any]) -> str:
    """Convert the upstream JSONs into a natural-language strategy description
    that the code-generation LLM can follow."""
    tp = upstream.get("theme_pick", {})
    fe = upstream.get("factor_extract", {})

    theme = tp.get("theme", "Unknown")
    confidence = tp.get("confidence", 0)
    rationale = tp.get("rationale", "")
    purpose = fe.get("purpose", "")
    logic = fe.get("computation_logic", "")

    return textwrap.dedent(f"""\
        Theme: {theme} (confidence: {confidence})
        Rationale: {rationale}

        Factor Purpose:
        {purpose}

        Computation Logic (from the paper):
        {logic}
    """)


def build_prompt(
    paper_name: str,
    upstream: Dict[str, Any],
    manifest: Dict[str, str],
    selected_tables: List[str],
) -> str:
    """Assemble the full code-generation prompt.

    Includes: strategy description (from upstream JSONs) + data catalog
    (real Sharadar file paths/sizes) + pandas structural pattern + rules.
    """
    strategy_desc = _build_strategy_description(upstream)

    # Build data file listing with paths, sizes, schemas
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
        Write a Python backtest that replicates the factor described below.
        Use ONLY the Sharadar tables listed.

        PAPER: {paper_name}

        STRATEGY / FACTOR DESCRIPTION:
        {strategy_desc}

        DATA FILES (real Sharadar data — use EXACT paths):
        {file_listing}

        {PANDAS_PATTERN}

        OUTPUT:
        - Save 'returns.csv' with columns: date (YYYY-MM-DD), strategy_returns (float)
        - Monthly frequency (one row per rebalance month)
        - Print "METRICS:" followed by JSON with at least:
          sharpe_ratio, annualized_return, total_return, max_drawdown, n_months
        - Also print "Backtest complete. Rows: N" and first/last 5 rows of returns

        RULES:
        - Use ONLY pandas, numpy, and json (standard library). All imports at top.
        - Use EXACT file paths shown above. Use usecols= for large files (SEP, DAILY).
        - Return ONLY executable Python code. No markdown fences. No explanations.
        - The code must be fully self-contained (no function calls to external modules).
    """)
    return prompt


SYSTEM_MSG = (
    "You are an expert quantitative finance Python developer. "
    "Write clean backtest code using pandas and numpy against real Sharadar data. "
    "Follow the REQUIRED CODE STRUCTURE in the prompt exactly. "
    "Return ONLY executable Python code. No markdown fences. No explanations."
)


# ═════════════════════════════════════════════════════════════════════════
# 3. generate_code — call the LLM, get raw Python back
# ═════════════════════════════════════════════════════════════════════════
def _get_llm(model_choice: str = DEFAULT_MODEL):
    """Return a LangChain-compatible LLM with .invoke().

    Explicit options:
      - claude  -> Anthropic API (langchain-anthropic)
      - openai  -> OpenAI API (langchain-openai)
    Otherwise:
      - Agents.llm_config.get_llm_by_name(model_choice)
      - fallback to Claude if available
    """
    choice = str(model_choice).lower()

    # 1) Explicit Claude selection
    if choice == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.2,
                max_tokens=8192,
            )
        except ImportError as e:
            raise RuntimeError(
                "model_choice='claude' requires langchain-anthropic. "
                "Install it (pip install langchain-anthropic) and set ANTHROPIC_API_KEY."
            ) from e

    # 2) Explicit OpenAI selection
    if choice == "openai":
        try:
            from langchain_openai import ChatOpenAI
            # Uses OPENAI_API_KEY by default; optionally set OPENAI_MODEL in env
            import os
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "8192")),
            )
        except ImportError as e:
            raise RuntimeError(
                "model_choice='openai' requires langchain-openai. "
                "Install it (pip install langchain-openai) and set OPENAI_API_KEY."
            ) from e

    # 3) Repo-defined LLMs (base/finllama/custom, etc.)
    if get_llm_by_name is not None:
        try:
            return get_llm_by_name(model_choice)
        except Exception as e:
            logger.warning(f"get_llm_by_name('{model_choice}') failed: {e}")

    # 4) Fallback to Claude only when NOT explicitly requested
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            max_tokens=8192,
        )
    except ImportError:
        pass

    raise RuntimeError(
        "No LLM backend available. Install langchain-anthropic or langchain-openai, "
        "or ensure Agents.llm_config is importable."
    )

def _strip_fences(text: str) -> str:
    """Remove ```python ... ``` wrappers if present."""
    text = text.strip()
    text = re.sub(r'^```(?:python)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


def generate_code(
    paper_name: str,
    upstream: Dict[str, Any],
    manifest: Dict[str, str],
    selected_tables: List[str],
    model_choice: str = DEFAULT_MODEL,
) -> str:
    """Call the LLM to generate backtest Python code.

    Returns the raw Python code string (not JSON-wrapped).
    """
    llm = _get_llm(model_choice)
    user_prompt = build_prompt(paper_name, upstream, manifest, selected_tables)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=SYSTEM_MSG),
            HumanMessage(content=user_prompt),
        ]
    except ImportError:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ]

    logger.info(f"[generate_code] Calling LLM ({model_choice}) for '{paper_name}'...")
    resp = llm.invoke(messages)

    raw_content = getattr(resp, "content", "") or ""
    if not raw_content and isinstance(resp, str):
        raw_content = resp

    code = _strip_fences(raw_content)
    logger.info(
        f"[generate_code] Generated {len(code)} chars "
        f"({len(code.splitlines())} lines)"
    )

    if not code.strip():
        raise ValueError("LLM returned empty code.")

    return code


# ═════════════════════════════════════════════════════════════════════════
# 4. run_sandbox — execute code via SandboxExecutor
# ═════════════════════════════════════════════════════════════════════════
def run_sandbox(
    code: str,
    llm=None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Execute *code* inside SandboxExecutor with LLM-assisted retries.

    Output files (returns.csv, etc.) remain in the sandbox temp dir and are
    uploaded to GCS by save_outputs() after execution completes.
    """
    try:
        from .sandbox_executor_retry import SandboxExecutor
    except ImportError:
        try:
            from .sandbox_executor import SandboxExecutor
        except ImportError:
            raise ImportError(
                "Cannot import SandboxExecutor. Ensure sandbox_executor_retry.py "
                "is on PYTHONPATH or in the working directory."
            )

    executor = SandboxExecutor(
        llm=llm,
        max_retries=max_retries,
        timeout=timeout,
        extra_packages=["pandas", "numpy"],
        keep_sandbox=True,  # keep so we can upload output files to GCS
    )

    logger.info(
        f"[run_sandbox] Executing ({len(code)} chars, "
        f"retries={max_retries}, timeout={timeout}s)..."
    )
    result = executor.run(code)
    logger.info(
        f"[run_sandbox] Done — success={result['success']}, "
        f"retries_used={result['retries_used']}"
    )
    return result


# ═════════════════════════════════════════════════════════════════════════
# 5. retry_loop — outer generate → execute → (re-generate)*
# ═════════════════════════════════════════════════════════════════════════
def retry_loop(
    paper_name: str,
    upstream: Dict[str, Any],
    manifest: Dict[str, str],
    selected_tables: List[str],
    model_choice: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    outer_retries: int = OUTER_RETRIES,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Full  generate → sandbox → (re-generate → sandbox)*  loop.

    INNER retries: SandboxExecutor diagnoses + fixes within one code block.
    OUTER retries: if inner retries exhausted, re-prompt LLM from scratch
    with error context for a completely new code generation.
    """
    llm = _get_llm(model_choice)
    all_attempts: List[Dict[str, Any]] = []

    for outer in range(1 + outer_retries):
        label = f"outer {outer + 1}/{1 + outer_retries}"
        logger.info(f"[retry_loop] {paper_name} — {label}")

        # ── Generate ─────────────────────────────────────────────────
        try:
            if outer == 0:
                code = generate_code(
                    paper_name, upstream, manifest, selected_tables, model_choice,
                )
            else:
                code = _regenerate_with_error(
                    paper_name, upstream, manifest, selected_tables,
                    all_attempts[-1], model_choice,
                )
        except Exception as e:
            logger.error(f"[retry_loop] Code generation failed ({label}): {e}")
            all_attempts.append({"generation_error": str(e)})
            continue

        if dry_run:
            logger.info("[retry_loop] --dry-run: skipping sandbox.")
            return _build_final_artifact(
                paper_name, model_choice, code, {}, all_attempts,
                status_override="dry_run",
            )

        # ── Execute ──────────────────────────────────────────────────
        sandbox_result = run_sandbox(
            code, llm=llm, max_retries=max_retries, timeout=timeout,
        )

        all_attempts.append({
            "code": code,
            "sandbox_result": _sanitise_sandbox_result(sandbox_result),
        })

        if sandbox_result["success"]:
            logger.info(f"[retry_loop] SUCCESS on {label}")
            return _build_final_artifact(
                paper_name, model_choice,
                sandbox_result.get("code_final", code),
                sandbox_result, all_attempts,
            )

        logger.warning(
            f"[retry_loop] FAILED {label} — "
            f"{len(sandbox_result['error_history'])} inner errors"
        )

    # All outer attempts exhausted
    last = all_attempts[-1] if all_attempts else {}
    last_sr = last.get("sandbox_result", {})
    return _build_final_artifact(
        paper_name, model_choice,
        last_sr.get("code_final", last.get("code", "")),
        last_sr, all_attempts,
    )


def _regenerate_with_error(
    paper_name: str,
    upstream: Dict[str, Any],
    manifest: Dict[str, str],
    selected_tables: List[str],
    prev_attempt: Dict[str, Any],
    model_choice: str,
) -> str:
    """Re-prompt the LLM with the original context + previous failure."""
    llm = _get_llm(model_choice)

    prev_code = prev_attempt.get("code", "")
    prev_sr = prev_attempt.get("sandbox_result", {})
    prev_errors = prev_sr.get("error_history", [])
    last_stderr = prev_errors[-1] if prev_errors else prev_sr.get("stderr", "")

    # Include the full original prompt context so the LLM has data paths etc.
    original_prompt = build_prompt(paper_name, upstream, manifest, selected_tables)

    regen_prompt = textwrap.dedent(f"""\
        Your previous backtest code for paper '{paper_name}' FAILED.

        PREVIOUS CODE (first 3000 chars):
        ```python
        {prev_code[:3000]}
        ```

        LAST ERROR:
        ```
        {last_stderr[:2000]}
        ```

        ORIGINAL TASK (for reference — use the same data paths and structure):
        {original_prompt}

        Write a COMPLETE NEW corrected Python backtest. Fix the issues above.
        Return ONLY executable Python code. No markdown. No explanations.
    """)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=SYSTEM_MSG),
            HumanMessage(content=regen_prompt),
        ]
    except ImportError:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": regen_prompt},
        ]

    logger.info("[regenerate] Re-prompting LLM with error context...")
    resp = llm.invoke(messages)
    raw = getattr(resp, "content", "") or ""
    code = _strip_fences(raw)
    if not code.strip():
        raise ValueError("LLM returned empty code on regeneration.")
    return code


def _sanitise_sandbox_result(sr: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate large strings for JSON serialisation."""
    MAX = 5_000
    out = dict(sr)
    for key in ("stdout", "stderr", "code_final"):
        val = out.get(key, "")
        if isinstance(val, str) and len(val) > MAX:
            out[key] = val[:MAX] + f"\n... (truncated, {len(val)} total)"
    if "error_history" in out:
        out["error_history"] = [
            (e[:2000] if isinstance(e, str) else e) for e in out["error_history"]
        ]
    return out


def _build_final_artifact(
    paper_name: str,
    model_choice: str,
    code_final: str,
    sandbox_result: Dict[str, Any],
    all_attempts: List[Dict[str, Any]],
    status_override: Optional[str] = None,
) -> Dict[str, Any]:
    sr = sandbox_result if isinstance(sandbox_result, dict) else {}
    status = status_override or ("success" if sr.get("success") else "failed")

    return {
        "paper_id": paper_name,
        "model_choice": model_choice,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_final": code_final,
        "sandbox": {
            "success": sr.get("success", False),
            "retries_used": sr.get("retries_used", 0),
            "metrics": sr.get("metrics"),
            "returns_csv": sr.get("returns_csv"),
            "output_files": sr.get("output_files", []),
            "sandbox_path": sr.get("sandbox_path", ""),
            "stdout": sr.get("stdout", ""),
            "stderr": sr.get("stderr", ""),
        },
        "total_attempts": len(all_attempts),
    }


# ═════════════════════════════════════════════════════════════════════════
# 6. save_outputs — write all artifacts to GCS
# ═════════════════════════════════════════════════════════════════════════
def save_outputs(artifact: Dict[str, Any]) -> str:
    """Persist the replication artifact and execution outputs to GCS.

    Uploads:
        1. The artifact JSON (metadata + code + sandbox results):
           gs://<OUTPUT_BUCKET>/replication_outputs/<paper_id>/<timestamp>.json

        2. The final Python code as a standalone file:
           gs://<OUTPUT_BUCKET>/replication_outputs/<paper_id>/<timestamp>_code.py

        3. Any execution output files (returns.csv, PNGs, etc.) from the sandbox:
           gs://<OUTPUT_BUCKET>/replication_outputs/<paper_id>/<timestamp>_<filename>

    Returns the GCS URI of the artifact JSON.
    """
    paper_id = artifact.get("paper_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    gcs_prefix = f"{REPLICATION_PREFIX}{paper_id}/{ts}"

    client = _get_gcs_client()
    if client is None:
        logger.error("[save_outputs] GCS client unavailable — cannot save.")
        return ""

    bucket = client.bucket(OUTPUT_BUCKET)

    # ── 1. Artifact JSON ─────────────────────────────────────────────
    json_blob_name = f"{gcs_prefix}.json"
    try:
        blob = bucket.blob(json_blob_name)
        blob.upload_from_string(
            json.dumps(artifact, indent=2, default=str),
            content_type="application/json",
        )
        artifact_uri = f"gs://{OUTPUT_BUCKET}/{json_blob_name}"
        logger.info(f"[save_outputs] Artifact: {artifact_uri}")
    except Exception as e:
        logger.error(f"[save_outputs] Artifact upload failed: {e}")
        return ""

    # ── 2. Standalone code file ──────────────────────────────────────
    code = artifact.get("code_final", "")
    if code:
        code_blob_name = f"{gcs_prefix}_code.py"
        try:
            blob = bucket.blob(code_blob_name)
            blob.upload_from_string(code, content_type="text/x-python")
            logger.info(f"[save_outputs] Code:     gs://{OUTPUT_BUCKET}/{code_blob_name}")
        except Exception as e:
            logger.warning(f"[save_outputs] Code upload failed: {e}")

    # ── 3. Execution output files (returns.csv, etc.) ────────────────
    sandbox = artifact.get("sandbox", {})
    sandbox_path = sandbox.get("sandbox_path", "")
    output_files = sandbox.get("output_files", [])

    # output_files from SandboxExecutor are paths inside the sandbox dir.
    # If sandbox_path is empty but we have a returns_csv, try that directly.
    files_to_upload: List[str] = []
    if output_files:
        files_to_upload = list(output_files)
    elif sandbox_path:
        # Scan sandbox dir for output files
        OUTPUT_EXTENSIONS = {".csv", ".png", ".jpg", ".json", ".html", ".xlsx"}
        sandbox_dir = Path(sandbox_path)
        if sandbox_dir.exists():
            for f in sandbox_dir.iterdir():
                if f.is_file() and f.suffix.lower() in OUTPUT_EXTENSIONS:
                    if f.name not in ("backtest.py", "input_data.csv"):
                        files_to_upload.append(str(f))

    for fpath in files_to_upload:
        fpath_obj = Path(fpath)
        if not fpath_obj.exists():
            continue
        fname = fpath_obj.name
        blob_name = f"{gcs_prefix}_{fname}"
        try:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(fpath_obj))
            logger.info(f"[save_outputs] Output:   gs://{OUTPUT_BUCKET}/{blob_name}")
        except Exception as e:
            logger.warning(f"[save_outputs] Upload of {fname} failed: {e}")

    # ── Clean up sandbox temp dir now that files are in GCS ──────────
    if sandbox_path:
        try:
            shutil.rmtree(sandbox_path, ignore_errors=True)
            logger.info(f"[save_outputs] Cleaned up sandbox: {sandbox_path}")
        except Exception:
            pass

    return artifact_uri


# ═════════════════════════════════════════════════════════════════════════
# Paper listing
# ═════════════════════════════════════════════════════════════════════════
def list_papers_with_upstream(model_filter: Optional[str] = None) -> List[str]:
    """List paper names that have upstream agent outputs in GCS."""
    paper_names: Set[str] = set()
    client = _get_gcs_client()
    if client is None:
        logger.error("Cannot list papers: GCS client unavailable.")
        return []

    try:
        bucket = client.bucket(OUTPUT_BUCKET)
        blobs = bucket.list_blobs(prefix="", delimiter="/")
        _ = [blob.name for blob in blobs] # Trigger the API call to populate blobs.prefixes
        for prefix_str in blobs.prefixes:
            folder = prefix_str.rstrip("/")
            # Skip our own replication output folder
            if folder == REPLICATION_PREFIX.rstrip("/"):
                continue
            if model_filter:
                check = list(bucket.list_blobs(
                    prefix=f"{folder}/{model_filter}/", max_results=1,
                ))
                if not check:
                    continue
            paper_names.add(folder)
    except Exception as e:
        logger.warning(f"GCS listing failed: {e}")

    return sorted(paper_names)


# ═════════════════════════════════════════════════════════════════════════
# Top-level orchestration
# ═════════════════════════════════════════════════════════════════════════
def process_single(
    paper_name: str,
    model_choice: str = DEFAULT_MODEL,
    upstream_model_choice: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    dry_run: bool = False,
    prompt_only: bool = False,
    data_dir: str = DATA_DIR,
) -> Dict[str, Any]:
    """Process one paper end-to-end."""
    logger.info("=" * 70)
    logger.info(f"Processing: {paper_name}  (model={upstream_model_choice})")
    logger.info("=" * 70)

    # ── Load upstream ────────────────────────────────────────────────
    upstream = load_inputs(paper_name, model_choice=upstream_model_choice)
    tp = upstream.get("theme_pick", {})
    fe = upstream.get("factor_extract", {})
    logger.info(f"  theme:   {tp.get('theme', '?')} (conf={tp.get('confidence', '?')})")
    logger.info(f"  purpose: {fe.get('purpose', '')[:120]}...")

    # ── Scan Sharadar data ───────────────────────────────────────────
    manifest = scan_data_dir(data_dir)
    if not manifest:
        raise FileNotFoundError(f"No Sharadar data found in {data_dir}")
    logger.info(f"  Found {len(manifest)} Sharadar tables on disk")

    # ── Select tables ────────────────────────────────────────────────
    selected = select_tables(upstream, manifest)
    logger.info(f"  Selected tables: {selected}")

    # ── Prompt-only mode ─────────────────────────────────────────────
    if prompt_only:
        prompt = build_prompt(paper_name, upstream, manifest, selected)
        print(f"\n{'─' * 70}\nSYSTEM:\n{'─' * 70}")
        print(SYSTEM_MSG)
        print(f"\n{'─' * 70}\nUSER PROMPT:\n{'─' * 70}")
        print(prompt)
        print(f"{'─' * 70}")
        est_tokens = len(prompt) // 4 + 200
        print(f"Prompt: {len(prompt)} chars (~{est_tokens} tokens)")
        return {"paper_id": paper_name, "status": "prompt_only"}

    # ── Generate + Execute ───────────────────────────────────────────
    artifact = retry_loop(
        paper_name, upstream, manifest, selected,
        model_choice=model_choice,
        max_retries=max_retries,
        timeout=timeout,
        dry_run=dry_run,
    )

    # ── Save to GCS ──────────────────────────────────────────────────
    gcs_uri = save_outputs(artifact)
    artifact["gcs_uri"] = gcs_uri

    # ── Summary ──────────────────────────────────────────────────────
    status = artifact.get("status", "unknown")
    sr = artifact.get("sandbox", {})
    logger.info(f"  Status:  {status}")
    if sr.get("metrics"):
        logger.info(f"  Metrics: {json.dumps(sr['metrics'], indent=2)}")
    logger.info(f"  GCS:     {gcs_uri}")

    return artifact


def process_all(
    model_choice: str = DEFAULT_MODEL,
    upstream_model_choice: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    start_index: int = 0,
    dry_run: bool = False,
    data_dir: str = DATA_DIR,
) -> List[Dict[str, Any]]:
    """Process every paper that has upstream outputs."""
    papers = list_papers_with_upstream(model_filter=upstream_model_choice)
    total = len(papers)
    logger.info(f"Found {total} papers (model={upstream_model_choice})")

    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if start_index > total:
        raise ValueError(f"--start-index ({start_index}) > papers ({total})")
    if start_index > 0:
        logger.info(f"Skipping first {start_index} papers.")

    all_results: List[Dict[str, Any]] = []

    for i, paper_name in enumerate(papers[start_index:], start=start_index):
        logger.info(f"\n[{i + 1}/{total}] {paper_name}")
        try:
            result = process_single(
                paper_name, model_choice=model_choice, upstream_model_choice=upstream_model_choice,
                max_retries=max_retries, timeout=timeout,
                dry_run=dry_run, data_dir=data_dir,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed on {paper_name}: {e}")
            all_results.append({
                "paper_id": paper_name, "model_choice": model_choice,
                "status": "error", "error": str(e),
            })

    logger.info(f"Batch complete: {len(all_results)} papers processed.")
    return all_results


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and execute replication backtests from upstream agent outputs "
            "against real Sharadar data."
        ),
    )
    parser.add_argument(
        "--paper", type=str, default=None,
        help="Single paper name (e.g. 'ref_Abarbanell_and_Bushee_1998').",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all papers with upstream outputs.",
    )
    parser.add_argument(
        "--model",
        choices=["openai", "base", "finllama", "custom", "claude"],
        default=DEFAULT_MODEL,
        help=f"LLM for generation + sandbox fixing (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--upstream-model",
        choices=["openai", "base", "finllama", "custom"],
        default=None,
        help="Which upstream folder to read JSON from (paper/<upstream-model>/...). Defaults to --model if not set.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"Max sandbox (inner) retries (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Sandbox timeout seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="0-based start index for --all mode.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR,
        help=f"Path to Sharadar CSV directory (default: {DATA_DIR}).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate code but skip execution.")
    parser.add_argument("--prompt-only", action="store_true",
                        help="Print prompt and exit.")
    parser.add_argument("--list-papers", action="store_true",
                        help="List papers with upstream outputs.")
    args = parser.parse_args()

    upstream_model = args.upstream_model or args.model # use upstream_model if provided else the model
    # now upstream_model will be used to filter out data from upstream and the model will be used for generation
   
    if args.list_papers:
        papers = list_papers_with_upstream(model_filter=upstream_model)
        print(f"\nPapers with upstream outputs ({len(papers)}, model={upstream_model}):\n")
        for i, p in enumerate(papers):
            print(f"  [{i:3d}] {p}")
        return

    if not args.paper and not args.all:
        parser.error("Specify --paper <name> or --all.")

    if args.paper:
        result = process_single(
            args.paper, upstream_model_choice=upstream_model, model_choice=args.model,
            max_retries=args.max_retries, timeout=args.timeout,
            dry_run=args.dry_run, prompt_only=args.prompt_only,
            data_dir=args.data_dir,
        )
        status = result.get("status", "unknown")
        print(f"\n{'=' * 70}")
        print(f"RESULT: {status.upper()}")
        sr = result.get("sandbox", {})
        if status == "success" and sr.get("metrics"):
            print(f"Metrics:\n{json.dumps(sr['metrics'], indent=2)}")
        if result.get("gcs_uri"):
            print(f"GCS: {result['gcs_uri']}")
        if status == "failed":
            print(f"Last error:\n{sr.get('stderr', '')[:500]}")
        print(f"{'=' * 70}")
        return

    if args.all:
        results = process_all(
            upstream_model_choice=upstream_model, model_choice=args.model,
            max_retries=args.max_retries, timeout=args.timeout,
            start_index=args.start_index,
            dry_run=args.dry_run,
            data_dir=args.data_dir,
        )
        successes = sum(1 for r in results if r.get("status") == "success")
        failures = sum(1 for r in results if r.get("status") == "failed")
        errors = sum(1 for r in results if r.get("status") == "error")
        print(f"\n{'=' * 70}")
        print(f"BATCH COMPLETE: {len(results)} papers")
        print(f"  Success: {successes}  |  Failed: {failures}  |  Error: {errors}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()