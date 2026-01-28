"""
Paper replication agent pipeline for the Autonomous Quantitative Investment Research project.

This LangGraph workflow:
1) Prompts the user to select an LLM
2) Selects and downloads a research paper from GCS
3) Runs a multi-agent (ReAct-style) "paper replication" pipeline:
   - Agent 1: paper triage (paper_type)
   - Agent 2: extract economic quantities (paper_vars)  [ReAct + paper snippet tool]
   - Agent 3: extract factor logic (paper_logic)        [ReAct + paper snippet tool]
   - Agent 3.25: build a Sharadar-agnostic PAPER_SPEC
   - Agent 4: map PAPER_SPEC quantities to Sharadar     [ReAct + Sharadar dictionary tool]
   - Agent 4.5: build implementable FACTOR_SPEC
   - Agent 5: generate per-ticker (time-series) feature code
   - Agent 6: generate cross-sectional transforms + factor composer code
4) Produces a single python module string (replication_code) that can be saved/executed by a human.

"""

# Load environment variables BEFORE any langchain/langgraph imports
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import json
import logging
import re
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool

from Agents.config import LoggingConfig
from Agents.nodes import AgentState, select_model_node, select_paper_node  # reuse existing nodes
from Agents.gcs_utils import ensure_auth, list_available_papers, get_storage_client
from Agents.config import GCSConfig
from Agents.llm_config import get_llm_by_name

## ---------------------------------------------------------------------
## API keys / credentials
## ---------------------------------------------------------------------

## Sharadar / data access:
## This agent only *plans* data retrieval and generates code stubs. If you later add
## actual Sharadar access (e.g., GCS parquet reads, BigQuery, etc.), put credentials
## in .env and wire them into the execution layer, not the planning layer.
## ---------------------------------------------------------------------

# Setup logging
logging.basicConfig(
    level=LoggingConfig.LEVEL,
    format=LoggingConfig.FORMAT,
)
logger = logging.getLogger(LoggingConfig.LOGGER_NAME)

# ---------------------------------------------------------------------
# Context variables (tools access the current paper + dictionaries)
# ---------------------------------------------------------------------
_PAPER_TEXT: ContextVar[str] = ContextVar("_PAPER_TEXT", default="")
_SHARADAR_DD: ContextVar[Dict[str, Dict[str, str]]] = ContextVar("_SHARADAR_DD", default={})

# ---------------------------------------------------------------------
# Tooling (LangChain tools used by ReAct-style agents)
# ---------------------------------------------------------------------
@tool
def get_matching_content(key: str, max_matches: int = 10, window_lines: int = 2) -> List[str]:
    """Return snippets from the current paper text containing `key` (case-insensitive)."""
    paper_text = _PAPER_TEXT.get() or ""
    if not paper_text:
        return []

    lines = paper_text.splitlines()
    key_l = key.lower().strip()
    hits: List[str] = []
    for i, line in enumerate(lines):
        if key_l and key_l in line.lower():
            lo = max(0, i - window_lines)
            hi = min(len(lines), i + window_lines + 1)
            snippet = "\n".join(lines[lo:hi]).strip()
            if snippet:
                hits.append(snippet)
        if len(hits) >= max_matches:
            break
    return hits


@tool
def get_sh_matching_variables(keys: List[str], max_hits_per_key: int = 15) -> List[Dict[str, str]]:
    """Search Sharadar data dictionaries for the provided keywords."""
    dd = _SHARADAR_DD.get() or {}
    out: List[Dict[str, str]] = []

    lower_keys = [(k or "").lower().strip() for k in keys if (k or "").strip()]
    if not lower_keys:
        return out

    for table_name, table_dd in dd.items():
        for field, definition in table_dd.items():
            def_l = (definition or "").lower()
            for k in lower_keys:
                if k and k in def_l:
                    out.append(
                        {
                            "table": table_name,
                            "field": field,
                            "dd": definition,
                            "matched_on": k,
                        }
                    )
                    break
            if len(out) >= (max_hits_per_key * max(1, len(lower_keys))):
                return out
    return out



# ---------------------------------------------------------------------
# Paper loading helpers (GCS markdown)
# ---------------------------------------------------------------------
def _download_markdown_from_gcs_blob(storage_client, blob_name: str) -> str:
    """Download a markdown blob from the configured GCS bucket."""
    bucket = storage_client.bucket(GCSConfig.SOURCE_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def load_paper_markdown_from_gcs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure `paper_text` is populated with markdown from GCS.

    Supported inputs:
      - state['paper_gcs_uri'] = 'gs://<bucket>/<path>.md'
      - state['paper_blob_name'] = '<path>.md' (within the configured bucket)

    If neither is provided, this node is a no-op because `select_paper_node`
    already downloads the paper markdown from GCS based on `paper_name`.
    """
    paper_text = (state.get('paper_text') or '').strip()
    if paper_text and not paper_text.startswith('ERROR:'):
        return {}  # already loaded

    gcs_uri = (state.get('paper_gcs_uri') or '').strip()
    blob_name = (state.get('paper_blob_name') or '').strip()

    if not gcs_uri and not blob_name:
        # `select_paper_node` handles the normal case (folder-per-paper).
        return {}

    storage_client = get_storage_client()
    if storage_client is None:
        return {"paper_text": "ERROR: Failed to authenticate with GCS"}

    # Normalize inputs
    if gcs_uri:
        m = re.match(r"^gs://(?P<bucket>[^/]+)/(?P<path>.+)$", gcs_uri)
        if not m:
            return {"paper_text": f"ERROR: Invalid GCS URI: {gcs_uri}"}
        bucket_name = m.group('bucket')
        path = m.group('path')
        if bucket_name != GCSConfig.SOURCE_BUCKET_NAME:
            return {"paper_text": (
                "ERROR: GCS URI bucket does not match configured bucket. "
                "Update Agents.config.GCSConfig.SOURCE_BUCKET_NAME or pass a blob name."
            )}
        blob_name = path

    try:
        content = _download_markdown_from_gcs_blob(storage_client, blob_name)
    except Exception as e:
        return {"paper_text": f"ERROR: Could not download markdown from GCS ({blob_name}): {e}"}

    # Derive a friendly paper_name if possible
    derived_paper_name = state.get('paper_name') or ''
    if not derived_paper_name:
        # If stored under processed/<paper_name>/<file>.md, infer <paper_name>
        prefix = (GCSConfig.SOURCE_PREFIX or '').strip()
        if prefix and blob_name.startswith(prefix):
            remainder = blob_name[len(prefix):]
            derived_paper_name = remainder.split('/')[0] if remainder else ''
        else:
            derived_paper_name = blob_name

    return {"paper_name": derived_paper_name, "paper_text": content}

# ---------------------------------------------------------------------
# Prompts (from mickey_mouse_poc_paper)
# ---------------------------------------------------------------------
AGENT1_PROMPT = """You are an empirical finance paper triage agent.

TASK
Classify the paper based ONLY on the abstract below.

Return STRICT JSON (no markdown, no prose) with exactly:
- paper_type
- factor_family
- expected_frequency
- needs_full_paper
- uncertainties

ABSTRACT
{abstract}
"""

AGENT2_SYSTEM = """You are Agent 2: Variable Requirements Extractor for empirical finance replication.

You may call get_matching_content(key) to retrieve paper snippets.
You MUST call the tool at least 4 times before finalizing.

Extract ECONOMIC QUANTITIES only:
- No formulas
- No transforms (rank/zscore/winsorize/regression)
- No vendor/dataset/column names

Return ONLY JSON with:
{
  "quantities": [
    {"name": "...", "label": "...", "description": "...", "units": "return|ratio|usd|percent|indicator|score|other|null",
     "timing": null or "...", "paper_aliases": [], "depends_on": []}
  ],
  "universe_and_sample": {"filters": [], "breakpoints": "NYSE|all|other|null", "frequency": "daily|monthly|quarterly|unknown", "notes": "..."},
  "missing_or_unclear": [],
  "evidence": [{"quantity":"...", "keys_used":[...], "snippets":[...]}]
}"""

AGENT3_SYSTEM = """You are Agent 3: Factor Logic Extractor for empirical finance replication.

You may call get_matching_content(key) to retrieve paper snippets.
You MUST call the tool at least 6 times before finalizing.

MISSION
Extract HOW the factor is computed (logic), given the abstract and the already-extracted economic quantities.

HARD RULES
- Do NOT map to datasets/vendors/column names.
- Do NOT write Python code.
- Be explicit about windows, lags, skip periods, rebalancing frequency, and standardization.
- If the paper defines portfolio construction (sorts, long-short), extract that logic but do not compute returns.

OUTPUT: Return ONLY JSON with:
{
  "definitions": [{"name": "...", "definition_text": "..."}],
  "formula_steps": ["step 1 ...", "step 2 ...", ...],   # each step one sentence
  "lags_and_alignment": [{"quantity": "...", "rule": "..."}],
  "standardization": ["rank|zscore|winsorize|industry_neutralize|none|..."],
  "rebalancing": "daily|monthly|quarterly|unknown",
  "weighting": "equal|value|other|unknown",
  "portfolio_construction": {
      "method": "sorts|regression|signal_only|other",
      "details": "short text"
  },
  "missing_or_unclear": ["..."],
  "evidence": [{"topic":"...", "keys_used":[...], "snippets":[...]}]
}"""

AGENT325_SYSTEM = """You are Agent 3.25: PAPER_SPEC Planner for empirical finance replication.

MISSION
Given:
- paper_type (Agent 1)
- paper_vars (Agent 2)
- paper_logic (Agent 3)

Produce PAPER_SPEC: a clean, Sharadar-agnostic semantic plan describing WHAT the paper wants computed.

HARD RULES
- No tools. No paper searching. Use only provided inputs.
- Do NOT map to vendor/dataset/column names (Sharadar mapping happens later).
- Do NOT write Python code.
- Do NOT invent variables not supported by paper_vars / paper_logic.
- Use paper_type as a hard prior; keep the plan minimal.

OUTPUT: Return ONLY JSON with:
{
  "paper_meta": {
    "paper_type": "...",
    "factor_family": "...",
    "expected_frequency": "...",
    "doc": "1–2 sentences summary of what the paper's signal is."
  },
  "timing": {
    "rebalancing": "daily|monthly|quarterly|unknown",
    "information_cutoff_rule": "plain English rule for what info is allowed at signal date",
    "doc": "1 sentence."
  },
  "universe": {
    "filters": ["as stated or empty"],
    "breakpoints": "NYSE|all|other|null",
    "weighting": "equal|value|other|unknown",
    "doc": "1 sentence."
  },
  "economic_quantities": [
    {
      "name": "canonical_name",
      "doc": "paper-specific definition; no dataset words",
      "units": "return|ratio|usd|percent|indicator|score|other|null",
      "timing": "lag/window/frequency if stated else null",
      "notes": null or "short note"
    }
  ],
  "logic_outline": [
    "Step 1 ...",
    "Step 2 ...",
    "..."
  ],
  "open_questions": ["..."],
  "assumptions": ["..."]
}

QUALITY BAR
- Every economic quantity must have a docstring that would let a human implement it.
- logic_outline should be 5–12 steps, one sentence each."""

AGENT4_SYSTEM = """You are Agent 4: Sharadar Mapping Agent.

You map PAPER_SPEC economic quantities to Sharadar variables using ONLY the tool:
- get_sh_matching_variables(keys: list[str])

HARD RULES
- You MUST call get_sh_matching_variables at least 6 times before finalizing.
- Do NOT invent Sharadar variables. Use only tool results.
- Prefer matches in this order unless paper_spec indicates otherwise:
  DAILY (already point-in-time daily metrics) > SEP (prices/volume) > SF1 (fundamentals)
- For each economic quantity, decide:
  - status: one of ["exists_in_sharadar","derivable","missing","ambiguous"]
  - already_computed: true if Sharadar provides it directly as a field (e.g., DAILY.pb, DAILY.marketcap, etc.)
  - mapped_table: "SEP"|"SF1"|"DAILY"|null
  - mapped_field: variable name or null
  - selectors: keep empty {} unless the quantity explicitly requires a specific SF1 sub-type (if unknown, leave as {} and note ambiguity)
- If multiple plausible matches exist, set status="ambiguous" and list top 2–5 candidates.
- If nothing plausible exists, set status="missing".

TOOL SEARCH GUIDANCE
- Use multiple keyword strategies:
  (1) canonical quantity words: e.g., "market", "book", "equity", "profit", "return", "volatility", "volume"
  (2) paper wording: from quantity doc strings
  (3) synonyms: e.g., market cap, market capitalization; book value; enterprise value; dividends; accruals; liquidity; spread
- Keep keys short (1–2 words) and iterate.

OUTPUT: Return ONLY JSON with this exact shape:
{
  "required_keys": {
    "asset_id": "ticker",
    "date": "date",
    "notes": "If unknown, state what you assume"
  },
  "mappings": [
    {
      "quantity": "<paper_spec economic_quantities[i].name>",
      "doc": "<paper_spec economic_quantities[i].doc>",
      "status": "exists_in_sharadar|derivable|missing|ambiguous",
      "already_computed": true|false,
      "mapped_table": "SEP|SF1|DAILY|null",
      "mapped_field": "<variable>|null",
      "selectors": {},
      "candidates": [
        {"table":"SF1|SEP|DAILY", "field":"...", "why":"..."},
        {"table":"SF1|SEP|DAILY", "field":"...", "why":"..."}
      ],
      "notes": "<short>"
    }
  ],
  "hard_missing": ["<quantity names>"],
  "ambiguous_mappings": ["<quantity names>"]
}"""

AGENT45_SYSTEM = """You are Agent 4.5: FACTOR_SPEC Planner.

MISSION
Given:
- PAPER_SPEC (semantic plan)
- SHARADAR_MAPPINGS (what fields exist where)

Produce FACTOR_SPEC: final implementable plan.

CRITICAL RULE (MOST IMPORTANT)
Do NOT treat factor RETURNS (e.g., "Market Factor", "Size Factor", "UMO_1") as primitive Sharadar variables.
They are portfolio constructs. You MUST decompose them into computable ingredients.

DECOMPOSITION REQUIREMENTS (hard)
If the paper uses factor returns like Market/Size/Mispricing:
1) Create base ts_inputs that make factor-return construction possible:
   - stock prices -> stock_return (from SEP closeadj or close)
   - market capitalization -> value weights (from DAILY.marketcap if available)
2) Represent factor returns as portfolio-construction steps to be implemented later (Agent 6), NOT as ts_inputs.
3) Only use SHARADAR_MAPPINGS for selecting concrete fields (table+field). If mapping doesn't explicitly include returns,
   infer returns come from SEP closeadj/close (since SEP has prices) and document in assumptions.

TABLE/DATE CONVENTIONS
- SEP: date_col="date"
- DAILY: date_col="date"
- SF1: date_col="calendardate" (as-of alignment handled upstream)

OUTPUT MUST include non-empty ts_inputs whenever any factor return is needed.

OUTPUT JSON SHAPE (exact)
{
  "paper_meta": {...},
  "timing": {...},
  "universe": {...},

  "ts_inputs": [
    {"name": "...", "doc": "...", "table":"SEP|DAILY|SF1", "field":"...", "date_col":"date|calendardate", "notes": null}
  ],

  "ts_features": [
    {"name":"...", "doc":"...", "table":"SEP|DAILY|SF1", "date_col":"date|calendardate",
     "uses":["..."], "params":{...}, "output_name":"..."}
  ],

  "cs_transforms": [
    {"name":"...", "doc":"...", "applies_to":"...", "params":{...}, "output_name":"..."}
  ],

  "final_signal": {
    "name":"signal",
    "doc":"...",
    "expression":"symbolic expression using output_name identifiers",
    "notes": null
  },

  "portfolio_construction": [
    "Step-by-step instructions (1 sentence each) to build factor returns from stock_return + weights + sorts."
  ],

  "open_questions": [...],
  "assumptions": [...]
}

PLANNING LOGIC
- Use SHARADAR_MAPPINGS for concrete fields when available.
- If mappings say Market Factor missing, do NOT give up; instead:
  - add ts_input: stock_price (SEP.closeadj if present else SEP.close)
  - add ts_feature: stock_return (per ticker)
  - add ts_input: marketcap (DAILY.marketcap if present; else mark open_question)
  - add portfolio_construction steps for market return (value-weighted) and SMB (size sorts)
- Mispricing factors: if anomaly signals are not enumerated in PAPER_SPEC, add open_questions; still keep base ts_inputs for returns+weights.

Return JSON only."""

AGENT5_SYSTEM = """You are Agent 5: Time-Series Feature Code Generator (table-aware).

You receive FACTOR_SPEC which MUST include:
- ts_inputs: list of concrete passthrough inputs
    {name, doc, table, field, date_col, notes}
- ts_features: list of computed per-ticker features (optional)
    {name, doc, table, date_col, uses, params, output_name}

EXECUTION CONTRACT (LOCKED)
Each feature function is applied as:
    df.groupby("ticker", group_keys=False).apply(fn)

So each fn receives a single-ticker DataFrame and MUST:
- sort internally by date_col
- return a Series aligned to the input sub-DataFrame index

HARD RULES
- No cross-sectional ops (no groupby(date)).
- No joins.
- pandas/numpy only.
- Do NOT invent tables/fields/features not in FACTOR_SPEC.
- Output MUST be Python code only (no prose, no markdown).

OUTPUT CONTRACT (must define top-level objects)
TS_PLAN = list of dicts, each with:
  - name
  - kind: "passthrough"|"computed"
  - table: "SEP"|"DAILY"|"SF1"
  - date_col: "date"|"calendardate"
  - uses: list[str]
  - params: dict
  - output_name: str
  - source_field: str|null   (required for passthrough)

TS_FEATURES = dict[str, callable]

Callable signature:
def fn(ticker_df: pd.DataFrame, *, date_col: str, params: dict) -> pd.Series

IMPLEMENTATION REQUIREMENTS
- For each ts_input, generate a passthrough function that returns ticker_df[source_field].
  If missing, raise ValueError mentioning table + field.
- For each computed ts_feature, implement the described per-ticker logic using only columns in `uses`.
  If something is underspecified, raise ValueError with a short message."""

AGENT6_SYSTEM = """You are Agent 6: Cross-Sectional Transform + Final Factor Composer Code Generator.

INPUTS
You receive FACTOR_SPEC which includes:
- cs_transforms: per-date transforms to apply (rank/zscore/winsorize/etc.)
- final_signal: symbolic expression referencing named outputs
- portfolio_construction: optional step list (if factor returns are constructed)
- timing/universe metadata (for docstrings only)

EXECUTION CONTRACT (LOCKED)
- Cross-sectional transforms must be computed per date:
    panel.groupby("date", group_keys=False).apply(...)
- No per-ticker time-series computations here.
- No joins across tables here. Assume inputs already exist as columns on `panel`.

OUTPUT CONTRACT (must be Python code only; no prose/markdown)
Define top-level objects:

1) CS_PLAN: list of dicts with keys:
   - name
   - applies_to
   - params
   - output_name

2) CS_TRANSFORMS: dict[str, callable]
Each callable signature:
    def fn(panel: pd.DataFrame, feature_col: str, *, date_col: str, params: dict) -> pd.Series
Returns a Series aligned to panel rows.

3) compose_factor(panel: pd.DataFrame, *, date_col: str = "date", params: dict | None = None) -> pd.Series
- Applies CS_PLAN in order (creating columns on a working copy)
- Computes final signal from FACTOR_SPEC.final_signal.expression
- Returns the final signal Series aligned to panel rows

RULES
- Use only pandas and numpy.
- Implement robust zscore (handle zero std) and robust rank (pct rank).
- If a required column is missing on panel, raise ValueError naming the missing column.
- Keep code compact and readable.
- Use FACTOR_SPEC docstrings as function docstrings."""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _extract_abstract(paper_text: str) -> str:
    """Best-effort abstract extraction from a plain-text paper."""
    if not paper_text:
        return ""

    t = paper_text
    low = t.lower()

    # Common pattern: 'abstract' heading
    m = re.search(r"\babstract\b\s*\n+", low)
    if m:
        start = m.end()
        # stop at next common heading
        tail = t[start:]
        stop = re.search(r"\n\s*(?:1\.|introduction\b|keywords\b|j\.?e\.?l\.?\b)", tail.lower())
        if stop:
            return tail[: stop.start()].strip()
        return tail[:1500].strip()

    # Fallback: first N chars
    return t[:1500].strip()


def _extract_json(text: str) -> dict:
    """Parse a JSON object from an LLM response (defensive)."""
    if text is None:
        raise ValueError("Empty response from model")
    raw = text.strip()

    # Strip ``` fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    # Try direct load
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to locate first {...} block
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"Model did not return JSON. Raw output:\n{raw}")
    return json.loads(m.group(0))


def _load_sharadar_data_dicts() -> Dict[str, Dict[str, str]]:
    """
    Load Sharadar data dictionaries.
    """
    # ## CHANGE IF NEEDED:
    SH_DD_DIR = Path(__file__).parent / "data_dicts"  # Agents/data_dicts/
    tables = ["SF1", "DAILY", "SEP"]

    data_dicts: Dict[str, Dict[str, str]] = {}
    for table in tables:
        md_path = SH_DD_DIR / f"{table}.data_dict.md"
        if not md_path.exists():
            logger.warning(f"Sharadar data dictionary not found: {md_path}")
            data_dicts[table] = {}
            continue
        data_dicts[table] = _read_data_dictionary(md_path)

    return data_dicts


def _read_data_dictionary(md_path: Path) -> Dict[str, str]:
    """
    Parse a markdown data dictionary and return {indicator: definition}.

    Expected bullet format:
    - <b>indicator</b> : Name : [Section] Definition text...
    """
    text = md_path.read_text(encoding="utf-8")
    out: Dict[str, str] = {}

    bullet_re = re.compile(r"^\s*-\s*<b>(?P<ind>[^<]+)</b>\s*:(?P<rest>.*)$")
    for line in text.splitlines():
        m = bullet_re.match(line)
        if not m:
            continue
        ind = m.group("ind").strip()
        rest = m.group("rest").strip()
        if ind:
            out[ind] = rest
    return out


def _run_react_json_agent(
    *,
    llm,
    system_prompt: str,
    user_payload: Dict[str, Any],
    tools: List[Any],
    max_rounds: int = 14,
) -> Dict[str, Any]:
    """
    Minimal ReAct-style loop:
    - LLM may call tools; we execute and feed ToolMessage back.
    - Ends when the LLM returns a final JSON payload (no tool calls).
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: List[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    for _ in range(max_rounds):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args", {}) or {}
                tool_id = tc.get("id")

                if name not in tool_map:
                    raise ValueError(f"Unknown tool requested by model: {name}")

                result = tool_map[name].invoke(args)
                messages.append(
                    ToolMessage(
                        content=json.dumps({"tool": name, "args": args, "result": result}, default=str),
                        tool_call_id=tool_id,
                    )
                )
            continue

        # Final answer (should be JSON)
        return _extract_json(getattr(resp, "content", ""))

    raise RuntimeError("Exceeded max tool rounds without producing a final JSON answer.")


# ---------------------------------------------------------------------
# LangGraph nodes (multi-agent paper replication pipeline)
# ---------------------------------------------------------------------
def triage_paper_node(state: AgentState) -> AgentState:
    """Agent 1: classify paper type using the abstract."""
    paper_text = state.get("paper_text", "")
    abstract = _extract_abstract(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    prompt = AGENT1_PROMPT.format(abstract=abstract)

    try:
        resp = llm.invoke(prompt)
        paper_type = _extract_json(getattr(resp, "content", ""))
    except Exception as e:
        logger.error(f"Paper triage failed: {e}")
        paper_type = {
            "paper_type": "unknown",
            "factor_family": "unknown",
            "expected_frequency": "unknown",
            "needs_full_paper": True,
            "uncertainties": [str(e)],
        }

    return {
        **state,
        "abstract": abstract,
        "paper_type": paper_type,
    }


def agent2_extract_paper_vars_node(state: AgentState) -> AgentState:
    """Agent 2: extract economic quantities (ReAct + get_matching_content)."""
    paper_text = state.get("paper_text", "")
    _PAPER_TEXT.set(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    payload = {
        "paper_type": state.get("paper_type", {}),
        "abstract": state.get("abstract", ""),
    }

    paper_vars = _run_react_json_agent(
        llm=llm,
        system_prompt=AGENT2_SYSTEM,
        user_payload=payload,
        tools=[get_matching_content],
        max_rounds=12,
    )

    return {**state, "paper_vars": paper_vars}


def agent3_extract_factor_logic_node(state: AgentState) -> AgentState:
    """Agent 3: extract factor logic (ReAct + get_matching_content)."""
    paper_text = state.get("paper_text", "")
    _PAPER_TEXT.set(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    payload = {
        "paper_type": state.get("paper_type", {}),
        "paper_vars": state.get("paper_vars", {}),
        "abstract": state.get("abstract", ""),
    }

    paper_logic = _run_react_json_agent(
        llm=llm,
        system_prompt=AGENT3_SYSTEM,
        user_payload=payload,
        tools=[get_matching_content],
        max_rounds=14,
    )

    return {**state, "paper_logic": paper_logic}


def agent325_build_paper_spec_node(state: AgentState) -> AgentState:
    """Agent 3.25: build PAPER_SPEC (no tools)."""
    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    payload = {
        "paper_type": state.get("paper_type", {}),
        "paper_vars": state.get("paper_vars", {}),
        "paper_logic": state.get("paper_logic", {}),
    }

    resp = llm.invoke(
        [
            SystemMessage(AGENT325_SYSTEM),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )
    paper_spec = _extract_json(getattr(resp, "content", ""))

    return {**state, "paper_spec": paper_spec}


def agent4_map_to_sharadar_node(state: AgentState) -> AgentState:
    """Agent 4: map PAPER_SPEC to Sharadar (ReAct + get_sh_matching_variables)."""
    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    # Load dictionaries once per run
    dd = _load_sharadar_data_dicts()
    _SHARADAR_DD.set(dd)

    payload = {
        "paper_spec": state.get("paper_spec", {}),
        "notes": "Use only get_sh_matching_variables(keys) tool for mapping.",
    }

    sharadar_mappings = _run_react_json_agent(
        llm=llm,
        system_prompt=AGENT4_SYSTEM,
        user_payload=payload,
        tools=[get_sh_matching_variables],
        max_rounds=16,
    )

    return {**state, "sharadar_mappings": sharadar_mappings}


def agent45_build_factor_spec_node(state: AgentState) -> AgentState:
    """Agent 4.5: build FACTOR_SPEC (no tools)."""
    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    payload = {
        "paper_spec": state.get("paper_spec", {}),
        "sharadar_mappings": state.get("sharadar_mappings", {}),
    }

    resp = llm.invoke(
        [
            SystemMessage(AGENT45_SYSTEM),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )
    factor_spec = _extract_json(getattr(resp, "content", ""))

    return {**state, "factor_spec": factor_spec}


def agent5_codegen_ts_features_node(state: AgentState) -> AgentState:
    """Agent 5: generate TS_PLAN + TS_FEATURES python code."""
    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    factor_spec = state.get("factor_spec", {})
    resp = llm.invoke(
        [
            SystemMessage(AGENT5_SYSTEM),
            HumanMessage(content="FACTOR_SPEC:\n" + json.dumps(factor_spec, indent=2)),
        ]
    )
    ts_code = getattr(resp, "content", "") or ""

    # Validate required tokens
    for tok in ("TS_PLAN", "TS_FEATURES"):
        if tok not in ts_code:
            raise ValueError(f"Agent 5 output missing {tok}")

    return {**state, "ts_code": ts_code}


def agent6_codegen_cs_and_composer_node(state: AgentState) -> AgentState:
    """Agent 6: generate CS_PLAN + CS_TRANSFORMS + compose_factor python code."""
    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    factor_spec = state.get("factor_spec", {})
    resp = llm.invoke(
        [
            SystemMessage(AGENT6_SYSTEM),
            HumanMessage(content="FACTOR_SPEC:\n" + json.dumps(factor_spec, indent=2)),
        ]
    )
    cs_code = getattr(resp, "content", "") or ""

    for tok in ("CS_PLAN", "CS_TRANSFORMS", "def compose_factor"):
        if tok not in cs_code:
            raise ValueError(f"Agent 6 output missing required token: {tok}")

    return {**state, "cs_code": cs_code}


def finalize_replication_code_node(state: AgentState) -> AgentState:
    """Combine Agent 5 + Agent 6 outputs into a single, runnable module string."""
    ts_code = state.get("ts_code", "")
    cs_code = state.get("cs_code", "")

    module_code = (
        "\n".join(
            [
                '"""',
                "Auto-generated replication module (Agent 5 + Agent 6).",
                "",
                "IMPORTANT",
                "- This module is generated by an LLM planner. Review before executing.",
                "- Data loading / joins across Sharadar tables are intentionally NOT implemented here.",
                "  Build a panel with required columns, then call compose_factor(panel).",

                '"""',
                "",
                "import numpy as np",
                "import pandas as pd",
                "",
                "# -------------------------------------------------------------------",
                "# Agent 5 output: per-ticker time-series inputs/features",
                "# -------------------------------------------------------------------",
                ts_code.strip(),
                "",
                "# -------------------------------------------------------------------",
                "# Agent 6 output: cross-sectional transforms + final composer",
                "# -------------------------------------------------------------------",
                cs_code.strip(),
                "",
            ]
        )
        + "\n"
    )

    return {**state, "replication_code": module_code}


# ---------------------------------------------------------------------
# LangGraph builder
# ---------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow for the paper replication agent.

    Workflow:
        select_model -> select_paper -> load_paper_markdown -> triage_paper -> agent2_vars -> agent3_logic
        -> agent325_paper_spec -> agent4_sharadar_map -> agent45_factor_spec
        -> agent5_ts_codegen -> agent6_cs_codegen -> finalize -> END
    """
    graph = StateGraph(AgentState)

    # Reuse existing nodes (minimal change)
    graph.add_node("select_model", select_model_node)
    graph.add_node("select_paper", select_paper_node)
    graph.add_node("load_paper_markdown", load_paper_markdown_from_gcs_node)

    # Paper replication nodes
    graph.add_node("triage_paper", triage_paper_node)
    graph.add_node("agent2_vars", agent2_extract_paper_vars_node)
    graph.add_node("agent3_logic", agent3_extract_factor_logic_node)
    graph.add_node("agent325_paper_spec", agent325_build_paper_spec_node)
    graph.add_node("agent4_sharadar_map", agent4_map_to_sharadar_node)
    graph.add_node("agent45_factor_spec", agent45_build_factor_spec_node)
    graph.add_node("agent5_ts_codegen", agent5_codegen_ts_features_node)
    graph.add_node("agent6_cs_codegen", agent6_codegen_cs_and_composer_node)
    graph.add_node("finalize", finalize_replication_code_node)

    graph.set_entry_point("select_model")

    graph.add_edge("select_model", "select_paper")
    graph.add_edge("select_paper", "load_paper_markdown")
    graph.add_edge("load_paper_markdown", "triage_paper")
    graph.add_edge("triage_paper", "agent2_vars")
    graph.add_edge("agent2_vars", "agent3_logic")
    graph.add_edge("agent3_logic", "agent325_paper_spec")
    graph.add_edge("agent325_paper_spec", "agent4_sharadar_map")
    graph.add_edge("agent4_sharadar_map", "agent45_factor_spec")
    graph.add_edge("agent45_factor_spec", "agent5_ts_codegen")
    graph.add_edge("agent5_ts_codegen", "agent6_cs_codegen")
    graph.add_edge("agent6_cs_codegen", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Build the application
app = build_graph()


def main():
    """Run the agent workflow with default settings."""
    logger.info("=" * 80)
    logger.info("Autonomous Quantitative Investment Research - Paper Replication Agent")
    logger.info("=" * 80)

    # Authenticate with GCS and list available papers
    storage_client = ensure_auth()
    if storage_client:
        papers = list_available_papers(storage_client)
        logger.info(f"Found {len(papers)} available papers in GCS bucket")
        if papers:
            logger.info(f"First paper: {papers[0]}")
    else:
        logger.error("Failed to authenticate with GCS. Exiting.")
        return

    # Initialize workflow state
    initial_state = {
        "messages": [],
        "model_choice": "",  # leave empty to prompt user
        "paper_name": "",    # leave empty to auto-select first paper
        "paper_text": "",
        "paper_gcs_uri": "",   # optional: gs://<bucket>/<path>.md
        "paper_blob_name": "", # optional: <path>.md within configured bucket

        # Outputs of the replication pipeline
        "abstract": "",
        "paper_type": {},
        "paper_vars": {},
        "paper_logic": {},
        "paper_spec": {},
        "sharadar_mappings": {},
        "factor_spec": {},
        "ts_code": "",
        "cs_code": "",
        "replication_code": "",
    }

    logger.info("Starting workflow...")
    logger.info("-" * 80)

    result = app.invoke(initial_state)

    logger.info("=" * 80)
    logger.info("WORKFLOW COMPLETED")
    logger.info("=" * 80)

    print(f"\nPaper: {result.get('paper_name', 'N/A')}")
    print(f"\nModel Used: {result.get('model_choice', 'N/A')}")
    print("\nReplication code module:\n" + "-" * 80)
    print(result.get("replication_code", ""))


if __name__ == "__main__":
    main()