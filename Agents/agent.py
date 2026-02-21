"""
Paper replication agent pipeline for the Autonomous Quantitative Investment Research project.
(v2 — strict JSON contract, tool-call budgets, sanitized tool results)
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import json
import logging
import re
import httpx
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool

from Agents.config import LoggingConfig, LLMConfig
from Agents.nodes import AgentState, select_model_node, select_paper_node
from Agents.gcs_utils import ensure_auth, list_available_papers, get_storage_client, upload_agent_output
from Agents.config import GCSConfig
from Agents.llm_config import get_llm_by_name
from Agents.prompts import (
    AGENT1_PROMPT,
    AGENT2_EXTRACT_PROMPT,
    build_system_prompt,
)

logging.basicConfig(
    level=LoggingConfig.LEVEL,
    format=LoggingConfig.FORMAT,
)
logger = logging.getLogger(LoggingConfig.LOGGER_NAME)

MAX_SNIPPET_CHARS = 600
MAX_SNIPPETS_PER_CALL = 5

_PAPER_TEXT: ContextVar[str] = ContextVar("_PAPER_TEXT", default="")
_SHARADAR_DD: ContextVar[Dict[str, Dict[str, str]]] = ContextVar("_SHARADAR_DD", default={})


# ---------------------------------------------------------------------
# Tools
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
    bucket = storage_client.bucket(GCSConfig.SOURCE_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()


def load_paper_markdown_from_gcs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure `paper_text` is populated with markdown from GCS."""
    paper_text = (state.get('paper_text') or '').strip()
    if paper_text and not paper_text.startswith('ERROR:'):
        return {}

    gcs_uri = (state.get('paper_gcs_uri') or '').strip()
    blob_name = (state.get('paper_blob_name') or '').strip()

    if not gcs_uri and not blob_name:
        return {}

    storage_client = get_storage_client()
    if storage_client is None:
        return {"paper_text": "ERROR: Failed to authenticate with GCS"}

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

    derived_paper_name = state.get('paper_name') or ''
    if not derived_paper_name:
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
    if not paper_text:
        return ""

    t = paper_text
    low = t.lower()

    m = re.search(r"\babstract\b\s*\n+", low)
    if m:
        start = m.end()
        tail = t[start:]
        stop = re.search(r"\n\s*(?:1\.|introduction\b|keywords\b|j\.?e\.?l\.?\b)", tail.lower())
        if stop:
            return tail[: stop.start()].strip()
        return tail[:1500].strip()

    return t[:1500].strip()


# =====================================================================
# JSON EXTRACTION — v2: balanced-brace scanner, no ast.literal_eval
# =====================================================================
def _extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty response from model")

    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

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

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = candidate
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"//[^\n]*", "", repaired)
    repaired = re.sub(r"(?<=[\{,\s])\s*'([^']+)'\s*:", r' "\1":', repaired)
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
    repaired = re.sub(r"\bNaN\b", "null", repaired)
    repaired = re.sub(r"\bInfinity\b", "null", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"Could not parse JSON even after repair. Candidate:\n{candidate[:500]}"
    )


def sanitize_tool_result(result: Any, tool_name: str) -> str:
    try:
        if isinstance(result, list):
            cleaned = []
            for item in result[:MAX_SNIPPETS_PER_CALL]:
                s = str(item).strip()
                s = re.sub(r"Traceback \(most recent call last\).*", "", s, flags=re.S)
                s = re.sub(
                    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?(INFO|WARN|WARNING|ERROR|DEBUG).*?\n",
                    "", s,
                )
                if len(s) > MAX_SNIPPET_CHARS:
                    s = s[:MAX_SNIPPET_CHARS] + "..."
                s = s.strip()
                if s:
                    cleaned.append(s)
            payload = {"tool": tool_name, "result": cleaned}
        elif isinstance(result, dict):
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
    if messages:
        try:
            repair_messages = list(messages) + [
                HumanMessage(content=_REPAIR_PROMPT_FROM_CONTEXT)
            ]
            resp = llm.invoke(repair_messages)
            return _extract_json(getattr(resp, "content", "") or "")
        except (ValueError, Exception) as e:
            logger.warning(f"Context-based JSON repair failed: {e}")

    if broken_text and broken_text.strip():
        try:
            truncated = broken_text[:3000]
            resp = llm.invoke(_REPAIR_PROMPT_WITH_TEXT.format(text=truncated))
            return _extract_json(getattr(resp, "content", "") or "")
        except (ValueError, Exception) as e:
            logger.warning(f"Text-based JSON repair also failed: {e}")

    logger.error("All JSON repair strategies failed. Returning fallback error output.")
    return {
        "_error": "json_repair_failed",
        "_message": "Model could not produce valid JSON after multiple attempts.",
        "missing_or_unclear": ["entire output — model failed to produce valid JSON"],
    }


def _load_sharadar_data_dicts() -> Dict[str, Dict[str, str]]:
    SH_DD_DIR = Path(__file__).parent / "data_dicts"
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


def _coerce_tool_args(args: dict) -> dict:
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
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: List[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    total_tool_calls = 0
    json_retry_used = False
    last_content = ""

    for round_num in range(max_rounds):
        tools_available = total_tool_calls < max_total_tool_calls
        resp = (llm_with_tools if tools_available else llm).invoke(messages)

        finish_reason = (
            (getattr(resp, "response_metadata", None) or {}).get("finish_reason")
        )
        if finish_reason == "length":
            logger.warning(f"Round {round_num + 1}: Response truncated. Asking retry.")
            messages.append(resp)
            messages.append(
                HumanMessage(content="Your response was cut off. Be MORE CONCISE. Output ONLY the required JSON object.")
            )
            continue

        invalid_tcs = getattr(resp, "invalid_tool_calls", None) or []
        if invalid_tcs:
            logger.warning(f"Round {round_num + 1}: {len(invalid_tcs)} invalid tool call(s).")
            messages.append(resp)
            messages.append(
                HumanMessage(content=f"Malformed tool call. Available tools: {list(tool_map.keys())}. Use the correct format.")
            )
            continue

        tool_calls = getattr(resp, "tool_calls", None) or []

        if tool_calls:
            if len(tool_calls) > max_tool_calls_per_round:
                logger.warning(f"Round {round_num + 1}: {len(tool_calls)} tool calls requested; capping to {max_tool_calls_per_round}.")
                tool_calls = tool_calls[:max_tool_calls_per_round]

            total_tool_calls += len(tool_calls)

            if total_tool_calls >= max_total_tool_calls:
                logger.warning(f"Tool budget exhausted ({total_tool_calls}/{max_total_tool_calls}). Next round will force final JSON (tools disabled).")

            if len(tool_calls) > 1:
                logger.info(f"Round {round_num + 1}: {len(tool_calls)} tool calls. Processing sequentially: {[tc.get('name') for tc in tool_calls]}")

            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args", {}) or {}
                tool_id = tc.get("id")

                if isinstance(args, dict) and "properties" in args and isinstance(args["properties"], dict):
                    args = args["properties"]

                args = _coerce_tool_args(args)

                if name not in tool_map:
                    logger.warning(f"Round {round_num + 1}: Unknown tool '{name}'.")
                    messages.append(resp)
                    messages.append(
                        HumanMessage(content=f"Tool '{name}' does not exist. Available tools: {list(tool_map.keys())}.")
                    )
                    break

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

                try:
                    result = tool_map[name].invoke(args)
                except Exception as e:
                    logger.warning(f"Tool '{name}' raised {type(e).__name__}: {e}")
                    result = f"Tool error: {type(e).__name__}: {str(e)[:200]}"

                messages.append(
                    ToolMessage(content=sanitize_tool_result(result, name), tool_call_id=tool_id)
                )

            continue

        content = getattr(resp, "content", "") or ""
        last_content = content
        messages.append(resp)

        try:
            maybe_tc = json.loads(content)
            if isinstance(maybe_tc, dict) and "parameters" in maybe_tc and "name" not in maybe_tc:
                logger.warning(f"Round {round_num + 1}: Content looks like tool call missing 'name'.")
                messages.append(
                    HumanMessage(content=f"Your response looks like a tool call but is missing 'name'. Available tools: {list(tool_map.keys())}.")
                )
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            return _extract_json(content)
        except ValueError:
            if not json_retry_used:
                json_retry_used = True
                logger.warning(f"Round {round_num + 1}: Non-JSON final output. Requesting retry.")
                messages.append(
                    HumanMessage(content="Your output was not valid JSON. Return ONLY the JSON object matching the schema. No prose, no markdown.")
                )
                continue
            else:
                logger.warning(f"Round {round_num + 1}: Second JSON failure. Falling through to JSON repair.")
                return _force_json(llm, content, messages=messages)

    logger.warning(f"Exhausted {max_rounds} rounds without final JSON. Forcing final JSON output.")
    messages.append(
        HumanMessage(content="STOP. You have used all available rounds. Return ONLY the final JSON object now. No tools. No prose.")
    )
    resp = llm.invoke(messages)
    content = getattr(resp, "content", "") or ""
    try:
        return _extract_json(content)
    except ValueError:
        return _force_json(llm, content, messages=messages)


# ---------------------------------------------------------------------
# Endpoint validation
# ---------------------------------------------------------------------
def validate_endpoint_node(state: AgentState) -> AgentState:
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
        raise ValueError(f"Model '{model_name}' not found on vLLM. Available: {available}")

    logger.info(f"Endpoint OK — '{model_name}' is available.")
    return {}


# ---------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------
def _log_agent_output(agent_name: str, output: dict, max_chars: int = 3000):
    try:
        summary = json.dumps(output, indent=2, default=str)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n... (truncated)"
        logger.info(f"\n{'='*60}\n  {agent_name} OUTPUT\n{'='*60}\n{summary}\n{'='*60}")
    except Exception:
        logger.info(f"{agent_name} OUTPUT: (could not serialize)")


ALLOWED_THEMES = {
    "Profitability", "Low Leverage", "Accruals", "Investment",
    "Debt Issuance", "Size", "Quality", "Seasonality", "Value", "Momentum",
}


def triage_paper_node(state: AgentState) -> AgentState:
    """Agent 1: classify paper theme using the abstract."""
    paper_text = state.get("paper_text", "")
    abstract = _extract_abstract(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    prompt = build_system_prompt(AGENT1_PROMPT, max_tool_calls=0, abstract=abstract)

    try:
        resp = llm.invoke(prompt)
        theme_pick = _extract_json(getattr(resp, "content", ""))

        theme_pick.setdefault("confidence", 0.0)
        theme_pick["confidence"] = max(0.0, min(1.0, float(theme_pick["confidence"])))
        if theme_pick.get("theme") not in ALLOWED_THEMES:
            theme_pick["theme"] = "unknown"
            theme_pick["confidence"] = 0.0
        theme_pick.setdefault("rationale", "")
        theme_pick.setdefault("evidence_phrases", [])
        theme_pick["evidence_phrases"] = theme_pick["evidence_phrases"][:5]

    except Exception as e:
        logger.error(f"Paper triage failed: {e}")
        theme_pick = {
            "theme": "unknown",
            "confidence": 0.0,
            "rationale": str(e),
            "evidence_phrases": [],
        }

    _log_agent_output("Agent 1 (triage)", theme_pick)

    return {
        **state,
        "abstract": abstract,
        "theme_pick": theme_pick,
    }


def agent2_extract_factor_node(state: AgentState) -> AgentState:
    """Agent 2: extract factor purpose and computation logic."""
    paper_text = state.get("paper_text", "")
    _PAPER_TEXT.set(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    abstract = state.get("abstract", "")
    payload = {
        "theme_pick": state.get("theme_pick", {}),
        "abstract": abstract,
    }

    prompt = build_system_prompt(AGENT2_EXTRACT_PROMPT, max_tool_calls=5, abstract=abstract)

    try:
        factor_extract = _run_react_json_agent(
            llm=llm,
            system_prompt=prompt,
            user_payload=payload,
            tools=[get_matching_content],
            max_rounds=7,
            max_tool_calls_per_round=2,
            max_total_tool_calls=5,
        )
        factor_extract.setdefault("purpose", "")
        factor_extract.setdefault("computation_logic", "")
    except Exception as e:
        logger.error(f"Factor extraction failed: {e}")
        factor_extract = {
            "purpose": "",
            "computation_logic": "",
        }

    _log_agent_output("Agent 2 (factor_extract)", factor_extract)

    return {**state, "factor_extract": factor_extract}


def save_output_to_gcs_node(state: AgentState) -> AgentState:
    """Save final agent outputs to the GCS output bucket as JSON."""
    paper_name = state.get("paper_name", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    blob_name = f"{paper_name}/{timestamp}.json"

    payload = {
        "paper_name": paper_name,
        "model_choice": state.get("model_choice", ""),
        "theme_pick": state.get("theme_pick", {}),
        "factor_extract": state.get("factor_extract", {}),
    }

    storage_client = get_storage_client()
    if storage_client is None:
        logger.warning("save_output_to_gcs: no GCS client available, skipping upload.")
        return {}

    try:
        uri = upload_agent_output(storage_client, blob_name, payload)
        logger.info(f"Agent output saved to {uri}")
        return {"output_gcs_uri": uri}
    except Exception as e:
        logger.warning(f"save_output_to_gcs: upload failed ({e}), continuing.")
        return {}


# ---------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------
def _paper_already_loaded(state: AgentState) -> str:
    """Route: skip paper loading if text is already provided."""
    if (state.get("paper_text") or "").strip():
        return "already_loaded"
    return "needs_loading"


# ---------------------------------------------------------------------
# LangGraph builder
# ---------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("select_model", select_model_node)
    graph.add_node("validate_endpoint", validate_endpoint_node)
    graph.add_node("select_paper", select_paper_node)
    graph.add_node("load_paper_markdown", load_paper_markdown_from_gcs_node)
    graph.add_node("triage_paper", triage_paper_node)
    graph.add_node("agent2_extract_factor", agent2_extract_factor_node)
    graph.add_node("save_output_to_gcs", save_output_to_gcs_node)

    graph.set_entry_point("select_model")

    graph.add_edge("select_model", "validate_endpoint")

    graph.add_conditional_edges(
        "select_model",
        _paper_already_loaded,
        {
            "already_loaded": "triage_paper",
            "needs_loading": "select_paper",
        },
    )

    graph.add_edge("select_paper", "load_paper_markdown")
    graph.add_edge("load_paper_markdown", "triage_paper")
    graph.add_edge("triage_paper", "agent2_extract_factor")
    graph.add_edge("agent2_extract_factor", "save_output_to_gcs")
    graph.add_edge("save_output_to_gcs", END)

    return graph.compile()


app = build_graph()


def main():
    """Run the agent workflow interactively."""
    logger.info("=" * 80)
    logger.info("Autonomous Quantitative Investment Research - Paper Replication Agent")
    logger.info("=" * 80)

    storage_client = ensure_auth()
    if storage_client:
        papers = list_available_papers(storage_client)
        logger.info(f"Found {len(papers)} available papers in GCS bucket")
        if papers:
            logger.info(f"First paper: {papers[0]}")
    else:
        logger.error("Failed to authenticate with GCS. Exiting.")
        return

    initial_state = {
        "messages": [],
        "model_choice": "",
        "paper_name": "",
        "paper_text": "",
        "paper_gcs_uri": "",
        "paper_blob_name": "",
        "abstract": "",
        "theme_pick": {},
        "factor_extract": {},
    }

    logger.info("Starting workflow...")
    logger.info("-" * 80)

    result = app.invoke(initial_state)

    logger.info("=" * 80)
    logger.info("WORKFLOW COMPLETED")
    logger.info("=" * 80)

    print(f"\nPaper: {result.get('paper_name', 'N/A')}")
    print(f"\nModel Used: {result.get('model_choice', 'N/A')}")
    print(f"\nTheme: {json.dumps(result.get('theme_pick', {}), indent=2)}")
    print(f"\nFactor Extract: {json.dumps(result.get('factor_extract', {}), indent=2)}")
    print(f"\nOutput saved to: {result.get('output_gcs_uri', 'N/A')}")


if __name__ == "__main__":
    main()
