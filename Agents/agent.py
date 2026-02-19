"""
Paper replication agent pipeline for the Autonomous Quantitative Investment Research project.
(v2 — strict JSON contract, tool-call budgets, sanitized tool results)

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

Changes from v1:
- _extract_json: balanced-brace scanner replaces greedy regex; ast.literal_eval removed.
- sanitize_tool_result(): truncates snippets, strips tracebacks/log lines, caps count.
- _force_json: minimal 3-line repair prompt instead of re-sending full system prompt.
- _run_react_json_agent: one JSON retry before repair; tool-budget cutoff disables tools.
- Agent nodes: use build_system_prompt() with per-agent tool budgets; lowered max_rounds.
"""

# Load environment variables BEFORE any langchain/langgraph imports
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import json
import logging
import re
import httpx
from contextvars import ContextVar
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool

from Agents.config import LoggingConfig, LLMConfig
from Agents.nodes import AgentState, select_model_node, select_paper_node  # reuse existing nodes
from Agents.gcs_utils import ensure_auth, list_available_papers, get_storage_client
from Agents.config import GCSConfig
from Agents.llm_config import get_llm_by_name
from Agents.prompts import (
    AGENT1_PROMPT,
    AGENT2_SYSTEM,
    AGENT3_SYSTEM,
    AGENT325_SYSTEM,
    AGENT4_SYSTEM,
    AGENT45_SYSTEM,
    AGENT5_SYSTEM,
    AGENT6_SYSTEM,
    build_system_prompt,
)


# Setup logging
logging.basicConfig(
    level=LoggingConfig.LEVEL,
    format=LoggingConfig.FORMAT,
)
logger = logging.getLogger(LoggingConfig.LOGGER_NAME)

# ---------------------------------------------------------------------
# Constants for tool-result sanitization
# ---------------------------------------------------------------------
MAX_SNIPPET_CHARS = 600
MAX_SNIPPETS_PER_CALL = 5

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


# =====================================================================
# JSON EXTRACTION — v2: balanced-brace scanner, no ast.literal_eval
# =====================================================================
def _extract_json(text: str) -> dict:
    """
    Extract the first valid JSON object from LLM output.

    Strategy:
    1. Strip markdown code fences.
    2. Try direct json.loads (fast path).
    3. Find the first balanced { ... } block via brace counting.
    4. Attempt json.loads on that block.
    5. Apply minimal safe repairs (trailing commas, single quotes, JS comments).
    6. Raise ValueError if all attempts fail.  (No ast.literal_eval.)
    """
    if not text or not text.strip():
        raise ValueError("Empty response from model")

    raw = text.strip()

    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    raw = raw.strip()

    # Fast path: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find first balanced { ... } using brace counter
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. Raw output:\n{raw[:500]}")

    depth = 0
    candidate = None
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                break

    if candidate is None:
        raise ValueError(f"Unbalanced braces in model output. Raw:\n{raw[:500]}")

    # Try parsing the balanced candidate
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # --- Minimal safe repairs ---
    repaired = candidate
    # 1. Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # 2. Remove JavaScript-style comments
    repaired = re.sub(r"//[^\n]*", "", repaired)
    # 3. Replace single-quoted keys/values with double quotes
    repaired = re.sub(r"(?<=[\{,\s])\s*'([^']+)'\s*:", r' "\1":', repaired)
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
    # 4. Replace JS-style NaN/Infinity with null
    repaired = re.sub(r"\bNaN\b", "null", repaired)
    repaired = re.sub(r"\bInfinity\b", "null", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # All attempts failed
    raise ValueError(
        f"Could not parse JSON even after repair. Candidate:\n{candidate[:500]}"
    )


# =====================================================================
# TOOL-RESULT SANITIZATION — v2: truncate, strip tracebacks, cap count
# =====================================================================
def sanitize_tool_result(result: Any, tool_name: str) -> str:
    """
    Clean and truncate tool output before putting into a ToolMessage.

    Guarantees:
    - List results capped to MAX_SNIPPETS_PER_CALL items.
    - Each string truncated to MAX_SNIPPET_CHARS characters.
    - Stack traces and log lines stripped.
    - Always returns valid JSON string.
    """
    try:
        if isinstance(result, list):
            cleaned = []
            for item in result[:MAX_SNIPPETS_PER_CALL]:
                s = str(item).strip()
                # Strip stack traces
                s = re.sub(
                    r"Traceback \(most recent call last\).*",
                    "", s, flags=re.S,
                )
                # Strip common log-line patterns
                s = re.sub(
                    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?"
                    r"(INFO|WARN|WARNING|ERROR|DEBUG).*?\n",
                    "", s,
                )
                # Truncate
                if len(s) > MAX_SNIPPET_CHARS:
                    s = s[:MAX_SNIPPET_CHARS] + "..."
                s = s.strip()
                if s:
                    cleaned.append(s)
            payload = {"tool": tool_name, "result": cleaned}
        elif isinstance(result, dict):
            # For dict results (e.g., Sharadar tool), truncate string values
            truncated = {}
            for k, v in result.items():
                if isinstance(v, str) and len(v) > MAX_SNIPPET_CHARS:
                    truncated[k] = v[:MAX_SNIPPET_CHARS] + "..."
                else:
                    truncated[k] = v
            payload = {"tool": tool_name, "result": truncated}
        else:
            payload = {"tool": tool_name, "result": str(result)[:MAX_SNIPPET_CHARS]}

        return json.dumps(payload, default=str, ensure_ascii=False)

    except Exception:
        return json.dumps({
            "tool": tool_name,
            "result": "Tool result could not be serialized",
            "error": "sanitization_failed",
        })


# =====================================================================
# JSON REPAIR — v3: uses accumulated messages, never crashes pipeline
# =====================================================================
_REPAIR_PROMPT_WITH_TEXT = (
    "You previously produced text that is not valid JSON.\n"
    "Convert it to a single valid JSON object matching the required schema.\n"
    "Output ONLY the JSON object. No prose, no markdown, no code fences.\n\n"
    "TEXT TO FIX:\n{text}\n"
)

_REPAIR_PROMPT_FROM_CONTEXT = (
    "STOP. You must now return your final answer.\n"
    "Based on everything discussed above, return ONLY a single valid JSON object "
    "matching the required schema.\n"
    "No prose, no markdown, no code fences. Just the JSON."
)


def _force_json(llm, broken_text: str, messages: List[BaseMessage] = None) -> dict:
    """
    Last-resort JSON repair. Two strategies:

    1. If we have accumulated messages context, append a stern finalize
       instruction and re-invoke with full context (the model can see
       all tool results and prior attempts).
    2. If that fails or messages aren't available, try the minimal
       text-repair prompt.
    3. If everything fails, return a fallback error dict instead of crashing.
    """
    # Strategy 1: Use accumulated messages (preferred — model has full context)
    if messages:
        try:
            repair_messages = list(messages) + [
                HumanMessage(content=_REPAIR_PROMPT_FROM_CONTEXT)
            ]
            resp = llm.invoke(repair_messages)
            return _extract_json(getattr(resp, "content", "") or "")
        except (ValueError, Exception) as e:
            logger.warning(f"Context-based JSON repair failed: {e}")

    # Strategy 2: Text-based repair (minimal prompt)
    if broken_text and broken_text.strip():
        try:
            truncated = broken_text[:3000]
            resp = llm.invoke(_REPAIR_PROMPT_WITH_TEXT.format(text=truncated))
            return _extract_json(getattr(resp, "content", "") or "")
        except (ValueError, Exception) as e:
            logger.warning(f"Text-based JSON repair also failed: {e}")

    # Strategy 3: Return a fallback error dict (never crash the pipeline)
    logger.error(
        "All JSON repair strategies failed. Returning fallback error output."
    )
    return {
        "_error": "json_repair_failed",
        "_message": "Model could not produce valid JSON after multiple attempts.",
        "missing_or_unclear": ["entire output — model failed to produce valid JSON"],
    }


# =====================================================================
# Sharadar data dictionary loading
# =====================================================================
def _load_sharadar_data_dicts() -> Dict[str, Dict[str, str]]:
    """Load Sharadar data dictionaries from markdown files."""
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


# =====================================================================
# Tool-arg coercion — vLLM/LLaMA 3.1 workaround
# =====================================================================
def _coerce_tool_args(args: dict) -> dict:
    """
    Fix vLLM/LLaMA 3.1 tool-call arg serialization bugs.

    Known issue: List[str] parameters arrive as JSON *strings* instead of
    actual lists, e.g. keys='["operating cash flow", "revenue"]' rather
    than keys=["operating cash flow", "revenue"].

    This detects string values that look like JSON arrays/objects and
    parses them back into native Python types.
    """
    if not isinstance(args, dict):
        return args

    coerced = {}
    for k, v in args.items():
        if isinstance(v, str):
            stripped = v.strip()
            if (stripped.startswith("[") and stripped.endswith("]")) or \
               (stripped.startswith("{") and stripped.endswith("}")):
                try:
                    coerced[k] = json.loads(stripped)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
        coerced[k] = v
    return coerced


# =====================================================================
# ReAct JSON Agent Loop — v2: tightened budgets, sanitized tools,
#   one JSON retry before repair, tool-budget cutoff disables tools
# =====================================================================
def _run_react_json_agent(
    *,
    llm,
    system_prompt: str,
    user_payload: Dict[str, Any],
    tools: List[Any],
    max_rounds: int = 8,
    max_tool_calls_per_round: int = 3,
    max_total_tool_calls: int = 8,
) -> Dict[str, Any]:
    """
    Minimal ReAct-style loop with strict safety rails.

    Protocol:
    - LLM responds in one of two modes per round:
        MODE A: tool call(s) → we execute, feed ToolMessage back, loop.
        MODE B: final JSON → we parse, return.
    - Tool budget enforced: once exhausted, LLM invoked WITHOUT tools bound.
    - One JSON-retry allowed before falling through to _force_json repair.
    - LLaMA 3 / vLLM constraint: tool calls split into one-per-AIMessage.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: List[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    total_tool_calls = 0
    json_retry_used = False  # allow exactly one in-loop JSON retry
    last_content = ""

    for round_num in range(max_rounds):
        # Once tool budget is exhausted, invoke WITHOUT tools to force JSON output
        tools_available = total_tool_calls < max_total_tool_calls
        resp = (llm_with_tools if tools_available else llm).invoke(messages)

        # --- Handle truncated output (model hit max_tokens) ---
        finish_reason = (
            (getattr(resp, "response_metadata", None) or {}).get("finish_reason")
        )
        if finish_reason == "length":
            logger.warning(
                f"Round {round_num + 1}: Response truncated (finish_reason='length'). "
                "Asking model to retry concisely."
            )
            messages.append(resp)
            messages.append(
                HumanMessage(
                    content=(
                        "Your response was cut off. Be MORE CONCISE. "
                        "Output ONLY the required JSON object, nothing else."
                    )
                )
            )
            continue

        # --- Handle invalid / malformed tool calls ---
        invalid_tcs = getattr(resp, "invalid_tool_calls", None) or []
        if invalid_tcs:
            logger.warning(
                f"Round {round_num + 1}: {len(invalid_tcs)} invalid tool call(s). "
                "Asking model to retry."
            )
            messages.append(resp)
            messages.append(
                HumanMessage(
                    content=(
                        f"Malformed tool call. Available tools: {list(tool_map.keys())}. "
                        "Use the correct format and try again."
                    )
                )
            )
            continue

        tool_calls = getattr(resp, "tool_calls", None) or []

        # =============================================================
        # MODE A — Tool calls
        # =============================================================
        if tool_calls:
            # Cap per-round
            if len(tool_calls) > max_tool_calls_per_round:
                logger.warning(
                    f"Round {round_num + 1}: {len(tool_calls)} tool calls requested; "
                    f"capping to {max_tool_calls_per_round}."
                )
                tool_calls = tool_calls[:max_tool_calls_per_round]

            total_tool_calls += len(tool_calls)

            if total_tool_calls >= max_total_tool_calls:
                logger.warning(
                    f"Tool budget exhausted ({total_tool_calls}/{max_total_tool_calls}). "
                    "Next round will force final JSON (tools disabled)."
                )

            if len(tool_calls) > 1:
                logger.info(
                    f"Round {round_num + 1}: {len(tool_calls)} tool calls. "
                    f"Processing sequentially: {[tc.get('name') for tc in tool_calls]}"
                )

            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args", {}) or {}
                tool_id = tc.get("id")

                # LLaMA sometimes wraps args in "properties"
                if isinstance(args, dict) and "properties" in args and isinstance(args["properties"], dict):
                    args = args["properties"]

                # --- vLLM/LLaMA 3.1 arg coercion ---
                # vLLM often serializes List[str] params as a JSON *string*
                # e.g. keys='["a", "b"]' instead of keys=["a", "b"]
                # Detect and fix: any string value that looks like a JSON array → parse it.
                args = _coerce_tool_args(args)

                # Unknown tool → ask to retry
                if name not in tool_map:
                    logger.warning(
                        f"Round {round_num + 1}: Unknown tool '{name}'. "
                        f"Available: {list(tool_map.keys())}."
                    )
                    messages.append(resp)
                    messages.append(
                        HumanMessage(
                            content=(
                                f"Tool '{name}' does not exist. "
                                f"Available tools: {list(tool_map.keys())}. "
                                "Please use one of the available tools or return final JSON."
                            )
                        )
                    )
                    break  # break tool loop; next round will retry

                # IMPORTANT: one tool call per AIMessage (LLaMA 3 constraint)
                single_tool_ai_msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": name,
                        "args": args,
                        "id": tool_id,
                        "type": "tool_call",
                    }],
                )
                messages.append(single_tool_ai_msg)

                # Execute tool with exception handling
                try:
                    result = tool_map[name].invoke(args)
                except Exception as e:
                    logger.warning(f"Tool '{name}' raised {type(e).__name__}: {e}")
                    result = f"Tool error: {type(e).__name__}: {str(e)[:200]}"

                # Sanitize before feeding back to model
                messages.append(
                    ToolMessage(
                        content=sanitize_tool_result(result, name),
                        tool_call_id=tool_id,
                    )
                )

            continue  # next ReAct round

        # =============================================================
        # MODE B — Final JSON attempt (no tool calls in response)
        # =============================================================
        content = getattr(resp, "content", "") or ""
        last_content = content
        messages.append(resp)

        # Check if content looks like a malformed tool call
        try:
            maybe_tc = json.loads(content)
            if isinstance(maybe_tc, dict) and "parameters" in maybe_tc and "name" not in maybe_tc:
                logger.warning(
                    f"Round {round_num + 1}: Content looks like tool call missing 'name'. "
                    "Asking model to retry."
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "Your response looks like a tool call but is missing 'name'. "
                            f"Available tools: {list(tool_map.keys())}. "
                            "Either call a tool properly or return your final JSON."
                        )
                    )
                )
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to parse as final JSON
        try:
            return _extract_json(content)
        except ValueError:
            # Allow exactly ONE in-loop retry before repair
            if not json_retry_used:
                json_retry_used = True
                logger.warning(
                    f"Round {round_num + 1}: Non-JSON final output. "
                    "Requesting retry (1 retry allowed)."
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "Your output was not valid JSON. "
                            "Return ONLY the JSON object matching the schema. "
                            "No prose, no markdown, no code fences."
                        )
                    )
                )
                continue
            else:
                # Second failure → last-resort repair (with full context)
                logger.warning(
                    f"Round {round_num + 1}: Second JSON failure. "
                    "Falling through to JSON repair."
                )
                return _force_json(llm, content, messages=messages)

    # -----------------------------------------------------------------
    # Exhausted all rounds — force one final attempt
    # -----------------------------------------------------------------
    logger.warning(
        f"Exhausted {max_rounds} rounds without final JSON. "
        "Forcing final JSON output."
    )
    messages.append(
        HumanMessage(
            content=(
                "STOP. You have used all available rounds. "
                "Return ONLY the final JSON object now. No tools. No prose."
            )
        )
    )
    resp = llm.invoke(messages)  # base LLM, no tools
    content = getattr(resp, "content", "") or ""
    try:
        return _extract_json(content)
    except ValueError:
        return _force_json(llm, content, messages=messages)


# ---------------------------------------------------------------------
# Endpoint validation node (fail fast on misconfigured model)
# ---------------------------------------------------------------------
def validate_endpoint_node(state: AgentState) -> AgentState:
    """Check that the selected model is reachable on the vLLM endpoint."""
    model_choice = state.get("model_choice", "openai")

    if model_choice == "openai":
        logger.info("Using OpenAI — skipping endpoint validation.")
        return {}

    llm = get_llm_by_name(model_choice)
    model_name = getattr(llm, "model_name", "unknown")
    base_url = getattr(llm, "openai_api_base", LLMConfig.LAMBDA_ENDPOINT)

    models_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/v1/models"
    logger.info(f"Validating endpoint: model={model_name} url={models_url}")

    try:
        resp = httpx.get(
            models_url,
            headers={"Authorization": f"Bearer {LLMConfig.LAMBDA_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        available = [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        raise ConnectionError(f"Cannot reach vLLM at {models_url}: {e}")

    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' not found on vLLM. Available: {available}"
        )

    logger.info(f"Endpoint OK — '{model_name}' is available.")
    return {}


# ---------------------------------------------------------------------
# LangGraph nodes (multi-agent paper replication pipeline)
# ---------------------------------------------------------------------
def _log_agent_output(agent_name: str, output: dict, max_chars: int = 3000):
    """Log a compact summary of agent JSON output for terminal debugging."""
    try:
        summary = json.dumps(output, indent=2, default=str)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n... (truncated)"
        logger.info(
            f"\n{'='*60}\n"
            f"  {agent_name} OUTPUT\n"
            f"{'='*60}\n"
            f"{summary}\n"
            f"{'='*60}"
        )
    except Exception:
        logger.info(f"{agent_name} OUTPUT: (could not serialize)")


def triage_paper_node(state: AgentState) -> AgentState:
    """Agent 1: classify paper type using the abstract."""
    paper_text = state.get("paper_text", "")
    abstract = _extract_abstract(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    # Build prompt with JSON contract (no tools for Agent 1)
    prompt = build_system_prompt(AGENT1_PROMPT, max_tool_calls=0, abstract=abstract)

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

    _log_agent_output("Agent 1 (triage)", paper_type)

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
        system_prompt=build_system_prompt(AGENT2_SYSTEM, max_tool_calls=6),
        user_payload=payload,
        tools=[get_matching_content],
        max_rounds=8,
        max_tool_calls_per_round=3,
        max_total_tool_calls=6,
    )

    _log_agent_output("Agent 2 (paper_vars)", paper_vars)

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
        system_prompt=build_system_prompt(AGENT3_SYSTEM, max_tool_calls=8),
        user_payload=payload,
        tools=[get_matching_content],
        max_rounds=10,
        max_tool_calls_per_round=3,
        max_total_tool_calls=8,
    )

    _log_agent_output("Agent 3 (paper_logic)", paper_logic)

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

    # Agent 3.25 has no tools — just a single LLM call with JSON contract
    system_prompt = build_system_prompt(AGENT325_SYSTEM, max_tool_calls=0)

    agent_messages = [
        SystemMessage(system_prompt),
        HumanMessage(content=json.dumps(payload, indent=2)),
    ]

    resp = llm.invoke(agent_messages)
    content = getattr(resp, "content", "") or ""

    try:
        paper_spec = _extract_json(content)
    except ValueError:
        logger.warning("Agent 3.25: JSON extraction failed; attempting repair.")
        agent_messages.append(resp)
        paper_spec = _force_json(llm, content, messages=agent_messages)

    _log_agent_output("Agent 3.25 (paper_spec)", paper_spec)

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
    }

    sharadar_mappings = _run_react_json_agent(
        llm=llm,
        system_prompt=build_system_prompt(AGENT4_SYSTEM, max_tool_calls=10),
        user_payload=payload,
        tools=[get_sh_matching_variables],
        max_rounds=12,
        max_tool_calls_per_round=3,
        max_total_tool_calls=10,
    )

    _log_agent_output("Agent 4 (sharadar_mappings)", sharadar_mappings)

    return {**state, "sharadar_mappings": sharadar_mappings}


# ---------------------------------------------------------------------
# LangGraph builder
# ---------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow for the paper replication agent.

    Workflow:
        select_model -> [validate_endpoint, select_paper -> load_paper_markdown] -> triage_paper
        -> agent2_vars -> agent3_logic -> agent325_paper_spec -> agent4_sharadar_map
        -> agent45_factor_spec -> agent5_ts_codegen -> agent6_cs_codegen -> finalize -> END
    """
    graph = StateGraph(AgentState)

    # Reuse existing nodes (minimal change)
    graph.add_node("select_model", select_model_node)
    graph.add_node("validate_endpoint", validate_endpoint_node)
    graph.add_node("select_paper", select_paper_node)
    graph.add_node("load_paper_markdown", load_paper_markdown_from_gcs_node)

    # Paper replication nodes
    graph.add_node("triage_paper", triage_paper_node)
    graph.add_node("agent2_vars", agent2_extract_paper_vars_node)
    graph.add_node("agent3_logic", agent3_extract_factor_logic_node)
    graph.add_node("agent325_paper_spec", agent325_build_paper_spec_node)
    graph.add_node("agent4_sharadar_map", agent4_map_to_sharadar_node)

    graph.set_entry_point("select_model")

    # Fan-out: validate endpoint in parallel with paper selection
    graph.add_edge("select_model", "select_paper")
    graph.add_edge("select_model", "validate_endpoint")
    graph.add_edge("select_paper", "load_paper_markdown")
    # Fan-in: both must complete before triage
    graph.add_edge("load_paper_markdown", "triage_paper")
    graph.add_edge("validate_endpoint", "triage_paper")
    graph.add_edge("triage_paper", "agent2_vars")
    graph.add_edge("agent2_vars", "agent3_logic")
    graph.add_edge("agent3_logic", "agent325_paper_spec")
    graph.add_edge("agent325_paper_spec", "agent4_sharadar_map")
    graph.add_edge("agent4_sharadar_map", END)

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
