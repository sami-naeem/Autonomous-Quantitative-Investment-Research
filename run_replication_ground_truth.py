"""
run_replication.py — Downstream replication-code generator & sandbox executor.

Supports two input sources:
  1. GCS (original): Reads upstream Agent 1 (theme_pick) + Agent 2 (factor_extract)
     JSON outputs for each paper from GCS.
  2. Local JSON (new): Reads ground-truth purpose + computation_logic from local
     JSON files, matching on paper name key.

Then uses the factor description to build a strategy prompt, calls the LLM to
generate backtest code against real Sharadar data, runs the code inside
SandboxExecutor with LLM-assisted retry logic, and writes all artifacts back
to GCS and/or locally.

GCS layout consumed (upstream, --source gcs):
    gs://investment-research-output/<paper_name>/<model_choice>/<timestamp>.json

Local JSON file (--source local-json):
    eval/osp_eval_policy_v003.json
    A single JSON dict keyed by paper name (e.g. "ref_Abarbanell_and_Bushee_1998.pdf")
    with values containing factor_logic_in_paper.{purpose, computation_logic} or
    top-level {purpose, computation_logic}.

GCS layout produced (downstream — same folder):
    gs://investment-research-output/<paper_name>/<model_choice>/<timestamp>_replication.json
    gs://investment-research-output/<paper_name>/<model_choice>/<timestamp>_code.py
    gs://investment-research-output/<paper_name>/<model_choice>/<timestamp>_returns.csv

Usage:
    # Original GCS mode (default)
    python run_replication.py --paper ref_Abarbanell_and_Bushee_1998
    python run_replication.py --all --model finllama

    # Local JSON ground-truth mode
    python run_replication.py --source local-json --paper ref_Abarbanell_and_Bushee_1998
    python run_replication.py --source local-json --json-file eval/osp_eval_policy_v003.json --all
    python run_replication.py --source local-json --list-papers
    python run_replication.py --source local-json --paper ref_Abarbanell_and_Bushee_1998 --prompt-only
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
else:
    GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "")
    GCS_CREDENTIALS_PATH = os.getenv("GCS_CREDENTIALS_PATH", "")

# ── Input source ─────────────────────────────────────────────────────────
# "gcs"        — original mode: fetch upstream Agent 1+2 JSONs from GCS
# "local-json" — new mode: read ground-truth purpose+computation_logic
#                from local JSON files, matching on paper name key
INPUT_SOURCE = os.getenv("REPLICATION_INPUT_SOURCE", "gcs")  # "gcs" | "local-json"

# Single ground-truth JSON file when using local-json mode.
# Expected to be a dict keyed by paper name (e.g. "ref_Abarbanell_and_Bushee_1998.pdf")
# with values containing at minimum {purpose, computation_logic} (and optionally
# other fields like factor_logic_in_paper, clean_code, etc.).
LOCAL_JSON_FILE = os.getenv("LOCAL_JSON_FILE", "eval/osp_eval_policy_v003.json")

# ── GCS bucket and layout ────────────────────────────────────────────────
# All inputs and outputs live in the same bucket under:
#   <paper_name>/<model_choice>/
#
# Upstream agent JSONs (input):
#   ref_Abarbanell_and_Bushee_1998/finllama/2025-06-15T14-22-09.json
#
# Replication artifacts (output — written back to same folder):
#   ref_Abarbanell_and_Bushee_1998/finllama/2025-06-20T10-30-00_replication.json
#   ref_Abarbanell_and_Bushee_1998/finllama/2025-06-20T10-30-00_code.py
#   ref_Abarbanell_and_Bushee_1998/finllama/2025-06-20T10-30-00_returns.csv
OUTPUT_BUCKET = os.getenv("GCS_OUTPUT_BUCKET", "investment-research-output")

# Valid model choices (used for listing and prefix construction)
VALID_MODELS = ["base", "custom", "finllama", "openai", "anthropic", "gemini"]

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
def _paper_model_prefix(paper_name: str, model_choice: str) -> str:
    """Return the GCS prefix for a paper+model folder.

    e.g. "ref_Abarbanell_and_Bushee_1998/finllama/"
    """
    return f"{paper_name}/{model_choice}/"


def _find_latest_upstream_blob(
    client: storage.Client,
    paper_name: str,
    model_choice: str,
) -> Optional[storage.Blob]:
    """Find the most-recent upstream agent output JSON in a paper/model folder.

    Scans gs://<OUTPUT_BUCKET>/<paper_name>/<model_choice>/ for .json blobs
    that are NOT replication artifacts (i.e. excludes *_replication.json,
    *_code.py, *_returns.csv etc.). Picks the most recently created blob.
    """
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = _paper_model_prefix(paper_name, model_choice)

    blobs = list(bucket.list_blobs(prefix=prefix))
    # Filter to upstream JSONs only — exclude our own replication outputs
    upstream_blobs = [
        b for b in blobs
        if b.name.endswith(".json")
        and "_replication.json" not in b.name
        and "_code.py" not in b.name
    ]
    if not upstream_blobs:
        return None

    upstream_blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return upstream_blobs[0]


def load_inputs(
    paper_name: str,
    model_choice: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Load upstream agent outputs (theme_pick + factor_extract) from GCS.

    Downloads the most recently created JSON blob at:
        gs://<OUTPUT_BUCKET>/<paper_name>/<model_choice>/<timestamp>.json

    Returns dict with: paper_id, model_choice, theme_pick, factor_extract
    """
    client = _get_gcs_client()
    if client is None:
        raise ConnectionError(
            f"Cannot load inputs for '{paper_name}': GCS client unavailable."
        )

    blob = _find_latest_upstream_blob(client, paper_name, model_choice)
    if blob is None:
        raise FileNotFoundError(
            f"No upstream output in gs://{OUTPUT_BUCKET}/"
            f"{_paper_model_prefix(paper_name, model_choice)}"
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
# 1b. load_inputs_from_json — read ground-truth from local JSON files
# ═════════════════════════════════════════════════════════════════════════
def _load_local_json(json_file: str) -> Dict[str, Dict[str, Any]]:
    """Load the single ground-truth JSON file and normalise its keys.

    The file is a dict keyed by paper name (with or without .pdf suffix),
    e.g.:
        {
          "ref_Abarbanell_and_Bushee_1998.pdf": {
            "factor_logic_in_paper": {"purpose": ..., "computation_logic": ...},
            ...
          }
        }

    Returns: {paper_name_without_pdf: <entry dict with purpose/computation_logic at top level>}
    """
    fpath = Path(json_file)
    if not fpath.exists():
        raise FileNotFoundError(f"Local JSON file not found: {json_file}")

    with open(fpath) as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in {json_file}, got {type(data).__name__}")

    merged: Dict[str, Dict[str, Any]] = {}
    for raw_key, val in data.items():
        if not isinstance(val, dict):
            continue
        paper_key = raw_key.removesuffix(".pdf")

        # The JSON may nest purpose/computation_logic inside
        # "factor_logic_in_paper" (like the osp_eval_policy format).
        # Flatten so downstream code can always find them at top level.
        flp = val.get("factor_logic_in_paper", {})
        entry = {**val}  # shallow copy
        if isinstance(flp, dict):
            # Promote purpose/computation_logic from nested dict if not already at top level
            for field in ("purpose", "computation_logic", "inputs", "formulas",
                          "edge_cases_and_timing", "output_contract", "factor_name"):
                if field not in entry and field in flp:
                    entry[field] = flp[field]

        merged[paper_key] = entry

    return merged


def load_inputs_from_json(
    paper_name: str,
    json_file: str = LOCAL_JSON_FILE,
) -> Dict[str, Any]:
    """Load ground-truth purpose + computation_logic from a local JSON file.

    Looks through the JSON file for a key matching *paper_name*
    (with or without .pdf suffix).

    Returns a dict shaped like the GCS upstream format so downstream code
    (build_prompt, select_tables, etc.) works unchanged:
        {
            "paper_id": ...,
            "model_choice": "ground_truth",
            "theme_pick": {"theme": <inferred>, "confidence": 1.0, "rationale": ""},
            "factor_extract": {"purpose": ..., "computation_logic": ...}
        }
    """
    all_papers = _load_local_json(json_file)

    # Try exact match first, then with .pdf suffix
    entry = all_papers.get(paper_name) or all_papers.get(paper_name + ".pdf")
    if entry is None:
        # Fuzzy: try stripping/adding "ref_" prefix
        alt = paper_name.removeprefix("ref_")
        entry = all_papers.get(alt) or all_papers.get("ref_" + paper_name)

    if entry is None:
        available = list(all_papers.keys())[:20]
        raise FileNotFoundError(
            f"Paper '{paper_name}' not found in '{json_file}'. "
            f"Available keys (first 20): {available}"
        )

    purpose = entry.get("purpose", "")
    logic = entry.get("computation_logic", "")

    # Attempt to infer a theme from purpose/logic text for table selection
    theme = _infer_theme_from_text(f"{purpose} {logic}")

    return {
        "paper_id": paper_name,
        "model_choice": "ground_truth",
        "theme_pick": {
            "theme": theme,
            "confidence": 1.0,
            "rationale": "Ground-truth from local JSON",
            "evidence_phrases": [],
        },
        "factor_extract": {
            "purpose": purpose,
            "computation_logic": logic,
            # Carry through any extra fields from the JSON entry
            **{k: v for k, v in entry.items()
               if k not in ("purpose", "computation_logic")},
        },
    }


def _infer_theme_from_text(text: str) -> str:
    """Best-effort theme inference from purpose/logic text for table selection."""
    t = text.lower()
    theme_keywords = {
        "Momentum":      ["momentum", "past return", "winner", "loser", "relative strength"],
        "Value":         ["value", "book-to-market", "earnings yield", "pe ratio", "pb ratio",
                          "undervalued", "overvalued"],
        "Quality":       ["quality", "accrual", "earnings quality", "fundamental signal"],
        "Profitability": ["profitability", "roe ", "roic", "gross margin", "net margin",
                          "operating profit"],
        "Investment":    ["investment", "capital expenditure", "capex", "asset growth"],
        "Size":          ["size", "small cap", "market capitaliz"],
        "Accruals":      ["accrual", "cash flow vs earnings", "operating accrual"],
        "Low Leverage":  ["leverage", "debt ratio", "debt-to-equity"],
        "Debt Issuance": ["debt issuance", "bond issue", "net debt"],
        "Seasonality":   ["seasonality", "january effect", "calendar"],
    }

    for theme, keywords in theme_keywords.items():
        if any(kw in t for kw in keywords):
            return theme
    return "Unknown"


def list_papers_from_json(json_file: str = LOCAL_JSON_FILE) -> List[str]:
    """List all paper names available in the local ground-truth JSON."""
    all_papers = _load_local_json(json_file)
    return sorted(all_papers.keys())


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
DATA_GUIDELINES = textwrap.dedent("""\
    DATA LOADING BEST PRACTICES:

    Universe construction:
      - Always filter SHARADAR_TICKERS to active US common stocks:
        table=='SEP', isdelisted=='N', category=='Domestic Common Stock'
      - Use the filtered ticker set to subset all other tables before processing

    Working with SEP (daily prices):
      - LARGE file — always use usecols=['ticker','date','closeadj'] (add others only if needed)
      - Use closeadj (split+dividend adjusted) for return calculations
      - Pivot to wide format (index=date, columns=ticker) for cross-sectional operations
        to avoid memory-exploding merges

    Working with SF1 (fundamentals):
      - Filter dimension=='ARQ' for quarterly data
      - Date column is 'calendardate' NOT 'date'

    CRITICAL WARNINGS:
      - NEVER merge large DataFrames on just 'ticker' without 'date' — causes cartesian explosion
      - NEVER use groupby().apply() on large datasets — use pivot + vectorized ops
      - Wide format (pivot_table) is strongly preferred for cross-sectional strategies
      - Avoid loading entire large CSVs into memory — use usecols and dtype where possible
      - Use .shift() for lagged signals to prevent look-ahead bias\
""")


def _build_strategy_description(upstream: Dict[str, Any]) -> str:
    """Convert the upstream JSONs into a natural-language strategy description
    that the code-generation LLM can follow."""
    tp = upstream.get("theme_pick", {})
    fe = upstream.get("factor_extract", {})

    theme = tp.get("theme", "Unknown")
    confidence = tp.get("confidence", 0)
    rationale = tp.get("rationale", "")
    evidence = tp.get("evidence_phrases", [])
    purpose = fe.get("purpose", "")
    logic = fe.get("computation_logic", "")

    evidence_str = ""
    if evidence:
        evidence_str = "\n    Evidence from paper:\n"
        for phrase in evidence[:5]:
            evidence_str += f"      - {phrase}\n"

    return textwrap.dedent(f"""\
        Theme: {theme} (confidence: {confidence})
        Rationale: {rationale}
        {evidence_str}
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
    (real Sharadar file paths/sizes) + general data guidelines + rules.
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
        Write a Python backtest that replicates the factor/strategy described below
        as faithfully as possible using the available Sharadar data.

        PAPER: {paper_name}

        STRATEGY / FACTOR DESCRIPTION:
        {strategy_desc}

        DATA FILES (real Sharadar data — use EXACT paths):
        {file_listing}

        {DATA_GUIDELINES}

        IMPLEMENTATION:
        - Study the factor description and identify the core signal, portfolio formation
          methodology, and holding period described in the paper.
        - Apply your knowledge of the empirical asset pricing literature to fill in
          any gaps: standard cross-sectional sorts, appropriate rebalance frequencies,
          and proper treatment of look-ahead bias and data availability lags.
        - Use the portfolio construction approach that most faithfully matches the paper.
        - If the paper uses decile/quantile sorts, implement long-short (top vs bottom).
          If the paper specifies a different methodology, follow that.
        - Use the rebalance frequency described in the paper.
        - Make reasonable simplifications where needed.

        OUTPUT REQUIREMENTS:
        - Save 'returns.csv' with columns: date (YYYY-MM-DD), strategy_returns (float)
        - The strategy_returns should represent the long-short spread return (or long-only
          if the paper is long-only) at the strategy's rebalance frequency
        - Print "Backtest complete. Rows: N" and first/last 5 rows of the returns

        RULES:
        - Use ONLY pandas, numpy, and json (standard library). All imports at top.
        - Use EXACT file paths shown above. Use usecols= for large files (SEP, DAILY).
        - Return ONLY executable Python code. No markdown fences. No explanations.
        - The code must be fully self-contained (no external module dependencies).
    """)
    return prompt


SYSTEM_MSG = (
    "You are a senior quantitative researcher replicating published factor strategies. "
    "You have deep expertise in cross-sectional asset pricing, Fama-French factor construction, "
    "portfolio sorts, and anomaly replication from academic finance papers. "
    "When implementing a factor, apply your knowledge of standard methodologies: "
    "Fama-MacBeth regressions, independent double sorts, NYSE breakpoints, "
    "value-weighted vs equal-weighted portfolio returns, and proper treatment of "
    "look-ahead bias, survivorship bias, and rebalancing conventions. "
    "If the paper's description is ambiguous, fall back on standard conventions from the "
    "empirical asset pricing literature (e.g., monthly rebalancing, quintile sorts, "
    "lagging fundamentals by at least one quarter). "
    "Write clean, self-contained Python code using pandas and numpy against real Sharadar data. "
    "Return ONLY executable Python code. No markdown fences. No explanations."
)


# ═════════════════════════════════════════════════════════════════════════
# 3. generate_code — call the LLM, get raw Python back
# ═════════════════════════════════════════════════════════════════════════
def _get_llm(model_choice: str = DEFAULT_MODEL):
    """Return a LangChain-compatible LLM with .invoke().

    Resolution order:
        1. Agents.llm_config.get_llm_by_name (handles all providers incl. frontier)
        2. Agents.provider_adapters for frontier providers (fallback if llm_config unavailable)
        3. langchain-anthropic hardcoded fallback
    """
    if get_llm_by_name is not None:
        try:
            return get_llm_by_name(model_choice)
        except Exception as e:
            logger.warning(f"get_llm_by_name('{model_choice}') failed: {e}")

    # Fallback: try frontier provider adapters directly
    try:
        from Agents.provider_adapters import is_frontier_provider, get_frontier_llm
        if is_frontier_provider(model_choice):
            return get_frontier_llm(model_choice, temperature=0.2, max_tokens=8192)
    except ImportError:
        pass

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
        "No LLM backend available. Install langchain-anthropic or ensure "
        "Agents.llm_config is importable."
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
        from sandbox_executor_retry import SandboxExecutor
    except ImportError:
        try:
            from sandbox_executor import SandboxExecutor
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

    Writes everything back into the same paper/model folder as the upstream
    agent outputs:

        gs://<OUTPUT_BUCKET>/<paper_id>/<model_choice>/<ts>_replication.json
        gs://<OUTPUT_BUCKET>/<paper_id>/<model_choice>/<ts>_code.py
        gs://<OUTPUT_BUCKET>/<paper_id>/<model_choice>/<ts>_returns.csv
        gs://<OUTPUT_BUCKET>/<paper_id>/<model_choice>/<ts>_<other_output>

    Returns the GCS URI of the artifact JSON.
    """
    paper_id = artifact.get("paper_id", "unknown")
    model_choice = artifact.get("model_choice", DEFAULT_MODEL)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    folder = _paper_model_prefix(paper_id, model_choice)  # e.g. ref_.../finllama/

    client = _get_gcs_client()
    if client is None:
        logger.error("[save_outputs] GCS client unavailable — cannot save.")
        return ""

    bucket = client.bucket(OUTPUT_BUCKET)

    # ── 1. Artifact JSON ─────────────────────────────────────────────
    json_blob_name = f"{folder}{ts}_replication.json"
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
        code_blob_name = f"{folder}{ts}_code.py"
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

    files_to_upload: List[str] = []
    if output_files:
        files_to_upload = list(output_files)
    elif sandbox_path:
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
        blob_name = f"{folder}{ts}_{fname}"
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
    """List paper names that have upstream agent outputs in GCS.

    Scans the bucket for top-level paper folders, then checks if a matching
    model subfolder exists with at least one upstream JSON.

    Layout: <paper_name>/<model_choice>/<timestamp>.json
    """
    client = _get_gcs_client()
    if client is None:
        logger.error("Cannot list papers: GCS client unavailable.")
        return []

    model = model_filter or DEFAULT_MODEL
    paper_names: Set[str] = set()

    try:
        bucket = client.bucket(OUTPUT_BUCKET)
        # List top-level "folders" (paper names)
        blobs_iter = bucket.list_blobs(delimiter="/")
        # Must consume the iterator to populate .prefixes
        list(blobs_iter)

        for prefix_str in blobs_iter.prefixes:
            paper_name = prefix_str.rstrip("/")
            # Check if this paper has a subfolder for the requested model
            # with at least one upstream JSON (exclude _replication.json)
            model_prefix = _paper_model_prefix(paper_name, model)
            model_blobs = list(bucket.list_blobs(prefix=model_prefix, max_results=10))
            has_upstream = any(
                b.name.endswith(".json") and "_replication.json" not in b.name
                for b in model_blobs
            )
            if has_upstream:
                paper_names.add(paper_name)
    except Exception as e:
        logger.warning(f"GCS listing failed: {e}")

    return sorted(paper_names)


# ═════════════════════════════════════════════════════════════════════════
# Top-level orchestration
# ═════════════════════════════════════════════════════════════════════════
def process_single(
    paper_name: str,
    model_choice: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    dry_run: bool = False,
    prompt_only: bool = False,
    data_dir: str = DATA_DIR,
    input_source: str = INPUT_SOURCE,
    json_file: str = LOCAL_JSON_FILE,
) -> Dict[str, Any]:
    """Process one paper end-to-end."""
    logger.info("=" * 70)
    logger.info(f"Processing: {paper_name}  (model={model_choice}, source={input_source})")
    logger.info("=" * 70)

    # ── Load upstream ────────────────────────────────────────────────
    if input_source == "local-json":
        upstream = load_inputs_from_json(paper_name, json_file=json_file)
    else:
        upstream = load_inputs(paper_name, model_choice=model_choice)

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
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    start_index: int = 0,
    dry_run: bool = False,
    data_dir: str = DATA_DIR,
    input_source: str = INPUT_SOURCE,
    json_file: str = LOCAL_JSON_FILE,
) -> List[Dict[str, Any]]:
    """Process every paper that has upstream outputs."""
    if input_source == "local-json":
        papers = list_papers_from_json(json_file=json_file)
    else:
        papers = list_papers_with_upstream(model_filter=model_choice)
    total = len(papers)
    logger.info(f"Found {total} papers (model={model_choice}, source={input_source})")

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
                paper_name, model_choice=model_choice,
                max_retries=max_retries, timeout=timeout,
                dry_run=dry_run, data_dir=data_dir,
                input_source=input_source, json_file=json_file,
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
        choices=["openai", "anthropic", "gemini", "base", "finllama", "custom"],
        default=DEFAULT_MODEL,
        help=f"LLM for generation + sandbox fixing (default: {DEFAULT_MODEL}).",
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
    parser.add_argument(
        "--source",
        choices=["gcs", "local-json"],
        default=INPUT_SOURCE,
        help=(
            "Input source for factor descriptions. "
            "'gcs' reads upstream agent outputs from GCS (original). "
            "'local-json' reads ground-truth purpose+computation_logic "
            f"from local JSON files (default: {INPUT_SOURCE})."
        ),
    )
    parser.add_argument(
        "--json-file", type=str, default=LOCAL_JSON_FILE,
        help=(
            "Path to ground-truth JSON file when using "
            f"--source local-json (default: {LOCAL_JSON_FILE})."
        ),
    )
    args = parser.parse_args()

    if args.list_papers:
        if args.source == "local-json":
            papers = list_papers_from_json(json_file=args.json_file)
            print(f"\nPapers in local JSON ({len(papers)}, file={args.json_file}):\n")
        else:
            papers = list_papers_with_upstream(model_filter=args.model)
            print(f"\nPapers with upstream outputs ({len(papers)}, model={args.model}):\n")
        for i, p in enumerate(papers):
            print(f"  [{i:3d}] {p}")
        return

    if not args.paper and not args.all:
        parser.error("Specify --paper <name> or --all.")

    if args.paper:
        result = process_single(
            args.paper, model_choice=args.model,
            max_retries=args.max_retries, timeout=args.timeout,
            dry_run=args.dry_run, prompt_only=args.prompt_only,
            data_dir=args.data_dir,
            input_source=args.source, json_file=args.json_file,
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
            model_choice=args.model,
            max_retries=args.max_retries, timeout=args.timeout,
            start_index=args.start_index,
            dry_run=args.dry_run,
            data_dir=args.data_dir,
            input_source=args.source, json_file=args.json_file,
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