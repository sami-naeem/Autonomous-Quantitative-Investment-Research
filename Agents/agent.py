"""
Paper replication agent pipeline for the Autonomous Quantitative Investment Research project.

v3 highlights:
- Strict JSON output contract for all agent stages.
- Tool-call budgets and sequential handling to prevent runaway loops.
- Sanitized tool results before reinjection into model context.
- Upgraded BM25-backed `get_matching_content` tool for paper passage retrieval.
- Hardened JSON extraction/repair for malformed wrapper output
  (for example doubled outer braces like {{ ... }} in repair responses).
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
from typing import Any, Dict, List, Set

from langgraph.graph import StateGraph, END

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool

from Agents.config import LoggingConfig, LLMConfig, BM25Config
from Agents.nodes import AgentState, select_model_node, select_paper_node
from Agents.gcs_utils import ensure_auth, list_available_papers, get_storage_client, upload_agent_output
from Agents.config import GCSConfig
from Agents.llm_config import get_llm_by_name
from Agents.prompts import (
    AGENT1_PROMPT,
    AGENT2_EXTRACT_PROMPT,
    build_system_prompt,
)
from Agents.paper_retrieval import chunk_academic_markdown, normalize_for_bm25
from Agents.factor_extract_fallback import (
    fallback_factor_extract_from_snippets,
    heuristic_factor_extract_from_snippets,
)

logging.basicConfig(
    level=LoggingConfig.LEVEL,
    format=LoggingConfig.FORMAT,
)
logger = logging.getLogger(LoggingConfig.LOGGER_NAME)

MAX_SNIPPET_CHARS = 900      # raised from 600 to fit academic paragraphs
MAX_SNIPPETS_PER_CALL = 5

_PAPER_TEXT: ContextVar[str] = ContextVar("_PAPER_TEXT", default="")
_SHARADAR_DD: ContextVar[Dict[str, Dict[str, str]]] = ContextVar("_SHARADAR_DD", default={})

# BM25 retrieval — populated at paper-load time, queried by the tool
_PAPER_CHUNKS: ContextVar[List[str]] = ContextVar("_PAPER_CHUNKS", default=[])
_PAPER_BM25: ContextVar[Any] = ContextVar("_PAPER_BM25", default=None)

BM25_SCORE_THRESHOLD = BM25Config.SCORE_THRESHOLD   # set via BM25_SCORE_THRESHOLD env var
BM25_TOP_K = BM25Config.TOP_K                       # set via BM25_TOP_K env var

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed — BM25 retrieval unavailable. Run: pip install rank-bm25")


# ---------------------------------------------------------------------
# Paper chunking & BM25 index helpers
# ---------------------------------------------------------------------
def _chunk_academic_markdown(text: str) -> List[str]:
    """Compatibility wrapper around the modular paper retrieval helper."""
    return chunk_academic_markdown(text)

def _build_paper_index(paper_text: str) -> None:
    """Build a BM25 index from the paper text and cache it in ContextVars.

    Called once per paper in agent2_extract_factor_node immediately after
    _PAPER_TEXT is set. Subsequent get_matching_content calls query the
    cached index without re-indexing.
    """
    if not _BM25_AVAILABLE:
        logger.error("[BM25] rank-bm25 not installed — index not built.")
        return

    chunks = _chunk_academic_markdown(paper_text)
    if not chunks:
        logger.warning("[BM25] No chunks extracted from paper text.")
        _PAPER_CHUNKS.set([])
        _PAPER_BM25.set(None)
        return

    tokenized = [normalize_for_bm25(c).lower().split() for c in chunks]
    index = BM25Okapi(tokenized)

    _PAPER_CHUNKS.set(chunks)
    _PAPER_BM25.set(index)

    avg_chars = sum(len(c) for c in chunks) / len(chunks)
    logger.info(
        f"[BM25] Index built: {len(chunks)} chunks | avg {avg_chars:.0f} chars/chunk"
    )
    # Log first 3 chunks so callers can verify chunking quality
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:300].replace('\n', '↵')
        logger.info(f"[BM25] Chunk[{i}]: {preview!r}")


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------
def _get_matching_content_legacy(key: str, max_matches: int = 10, window_lines: int = 2) -> List[str]:
    """(Legacy v1 — exact substring search over paper lines. Kept for comparison.)

    Original implementation: scans each line for an exact case-insensitive
    substring match and returns a ±window_lines context window around each hit.
    Replaced by the BM25-backed get_matching_content tool.
    """
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
def get_matching_content(query: str) -> List[str]:
    """Search the paper for passages relevant to a short keyword query.

    Use short, specific terms (1-3 words) — e.g. "book-to-market", "portfolio sort",
    "holding period", "lagged return", "regression coefficient".
    Avoid full sentences: the paper may phrase concepts differently from your query.
    Each call costs from your limited tool budget; use a distinct query per call.

    Returns up to 4 text passages (matched paragraphs + neighboring context).
    Returns a not-found message if no relevant passage exists — try a different keyword.
    """
    chunks = _PAPER_CHUNKS.get()
    bm25 = _PAPER_BM25.get()

    if not chunks or bm25 is None:
        if not _BM25_AVAILABLE:
            return ["rank-bm25 not installed. Cannot search paper."]
        return ["Paper index not available — index may not have been built yet."]

    query_l = query.lower().strip()
    if not query_l:
        return ["Empty query — provide a 1-3 word keyword."]

    scores = bm25.get_scores(query_l.split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Always log top-5 so we can see what BM25 thinks is relevant
    top5 = [(i, round(float(scores[i]), 3), chunks[i][:80].replace('\n', ' '))
            for i in ranked[:5]]
    logger.info(f"[BM25] query={query!r} | top-5 scores: {top5}")

    best_score = float(scores[ranked[0]]) if ranked else 0.0
    if best_score < BM25_SCORE_THRESHOLD:
        logger.info(
            f"[BM25] No strong match (best={best_score:.3f} < threshold={BM25_SCORE_THRESHOLD}). "
            f"Returning not-found for query={query!r}"
        )
        return [
            f"No relevant content found for '{query}' (best score={best_score:.2f}). "
            f"Try a shorter or different keyword."
        ]

    results: List[str] = []
    used: set = set()

    for idx in ranked[:BM25_TOP_K]:
        score = float(scores[idx])
        if score < BM25_SCORE_THRESHOLD or idx in used:
            break

        logger.info(
            f"[BM25] Returning chunk[{idx}] score={score:.3f}:\n{chunks[idx]}"
        )
        results.append(f"[MATCH score={score:.2f}]\n{chunks[idx]}")
        used.add(idx)

        # Add the following paragraph as context — papers elaborate forward
        next_idx = idx + 1
        if next_idx < len(chunks) and next_idx not in used:
            results.append(f"[CONTEXT]\n{chunks[next_idx]}")
            used.add(next_idx)

    return results if results else [f"No relevant content found for '{query}'."]


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
    if raw.startswith("{{") and raw.endswith("}}"):
        raw = raw[1:-1].strip()

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
    # Some models emit doubled outer braces: {{ ... }}. Unwrap if present.
    repaired = re.sub(r"^\s*\{\{", "{", repaired)
    repaired = re.sub(r"\}\}\s*$", "}", repaired)
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
    "Do not use doubled braces like {{ or }}.\n"
    "Output ONLY the JSON object. No prose, no markdown, no code fences.\n\n"
    "TEXT TO FIX:\n{text}\n"
)

_REPAIR_PROMPT_FROM_CONTEXT = (
    "STOP. You must now return your final answer.\n"
    "Based on everything discussed above, return ONLY a single valid JSON object "
    "matching the required schema.\n"
    "Do not use doubled braces like {{ or }}.\n"
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


def _normalize_tool_query(query: Any) -> str:
    text = str(query or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def _retrieved_snippet_score(snippet: str) -> float:
    text = (snippet or "").strip()
    if not text:
        return -1e9
    if "No relevant content found" in text:
        return -1000.0

    score = 0.0
    m = re.search(r"\[MATCH score=([0-9]+(?:\.[0-9]+)?)\]", text)
    if m:
        score += float(m.group(1)) * 10.0
    if text.startswith("[MATCH"):
        score += 5.0
    if text.startswith("[CONTEXT]"):
        score += 1.0
    score += min(len(text), 1500) / 1500.0
    return score


def _rank_snippets(snippets: List[str], max_items: int = 6, max_chars: int = 3500) -> List[str]:
    best_by_text: Dict[str, float] = {}
    for raw in snippets:
        text = str(raw or "").strip()
        if not text:
            continue
        s = _retrieved_snippet_score(text)
        prev = best_by_text.get(text)
        if prev is None or s > prev:
            best_by_text[text] = s

    ranked = sorted(best_by_text.items(), key=lambda kv: kv[1], reverse=True)
    out: List[str] = []
    total = 0
    for text, _ in ranked:
        if total + len(text) > max_chars:
            break
        out.append(text)
        total += len(text)
        if len(out) >= max_items:
            break
    return out


def _cached_evidence_blob(snippets: List[str], max_items: int = 6, max_chars: int = 3500) -> str:
    picked = _rank_snippets(snippets, max_items=max_items, max_chars=max_chars)
    return "\n\n---\n\n".join(picked)


def _collect_retrieval_snippets(tool_name: str, result: Any) -> List[str]:
    if tool_name != "get_matching_content" or not isinstance(result, list):
        return []
    out: List[str] = []
    for item in result:
        text = str(item or "").strip()
        if not text:
            continue
        if text.startswith("No relevant content found"):
            continue
        out.append(text)
    return out


def _run_react_json_agent(
    *,
    llm,
    system_prompt: str,
    user_payload: Dict[str, Any],
    tools: List[Any],
    max_rounds: int = 8,
    max_tool_calls_per_round: int = 3,
    max_total_tool_calls: int = 8,
    retrieved_snippets: List[str] | None = None,
    used_queries: Set[str] | None = None,
) -> Dict[str, Any]:
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    retrieved_snippets = retrieved_snippets if retrieved_snippets is not None else []
    used_queries = used_queries if used_queries is not None else set()

    messages: List[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    total_tool_calls = 0
    no_new_evidence_streak = 0
    json_retry_used = False
    last_content = ""

    def _push_cached_evidence_finalization(reason: str) -> None:
        evidence = _cached_evidence_blob(retrieved_snippets)
        evidence_text = evidence if evidence else "No cached evidence available."
        messages.append(
            HumanMessage(
                content=(
                    f"{reason}\n"
                    "Do not call tools again. Use ONLY the cached evidence below.\n\n"
                    f"Cached evidence:\n{evidence_text}\n\n"
                    "Return ONLY the final JSON object matching the schema. No prose."
                )
            )
        )

    for round_num in range(max_rounds):
        tools_available = total_tool_calls < max_total_tool_calls
        try:
            resp = (llm_with_tools if tools_available else llm).invoke(messages)
        except Exception as e:
            logger.warning(
                f"Round {round_num + 1}: LLM invoke failed ({type(e).__name__}: {e}). "
                "Falling back to forced final JSON path."
            )
            break

        finish_reason = (
            (getattr(resp, "response_metadata", None) or {}).get("finish_reason")
        )
        if finish_reason == "length":
            logger.warning(f"Round {round_num + 1}: Response truncated. Asking retry.")
            messages.append(AIMessage(content=getattr(resp, "content", "") or ""))
            messages.append(
                HumanMessage(content="Your response was cut off. Be MORE CONCISE. Output ONLY the required JSON object.")
            )
            continue

        invalid_tcs = getattr(resp, "invalid_tool_calls", None) or []
        if invalid_tcs:
            logger.warning(f"Round {round_num + 1}: {len(invalid_tcs)} invalid tool call(s).")
            messages.append(AIMessage(content=getattr(resp, "content", "") or ""))
            messages.append(
                HumanMessage(content=f"Malformed tool call. Available tools: {list(tool_map.keys())}. Use the correct format.")
            )
            continue

        tool_calls = getattr(resp, "tool_calls", None) or []

        if tool_calls:
            if len(tool_calls) > max_tool_calls_per_round:
                logger.warning(f"Round {round_num + 1}: {len(tool_calls)} tool calls requested; capping to {max_tool_calls_per_round}.")
                tool_calls = tool_calls[:max_tool_calls_per_round]

            if len(tool_calls) > 1:
                logger.info(f"Round {round_num + 1}: {len(tool_calls)} tool calls. Processing sequentially: {[tc.get('name') for tc in tool_calls]}")

            budget_exhausted_mid_round = False
            for tc in tool_calls:
                if total_tool_calls >= max_total_tool_calls:
                    logger.warning(
                        f"Tool budget exhausted ({total_tool_calls}/{max_total_tool_calls}). "
                        "Skipping remaining tool calls and forcing cached-evidence finalization."
                    )
                    _push_cached_evidence_finalization("Tool budget is exhausted.")
                    budget_exhausted_mid_round = True
                    break

                name = tc.get("name")
                args = tc.get("args", {}) or {}
                tool_id = tc.get("id")

                if isinstance(args, dict) and "properties" in args and isinstance(args["properties"], dict):
                    args = args["properties"]

                args = _coerce_tool_args(args)

                if name not in tool_map:
                    logger.warning(f"Round {round_num + 1}: Unknown tool '{name}'.")
                    # Never append raw tool-call responses here: some vLLM endpoints
                    # reject history that contains assistant messages with multiple
                    # tool_calls. Keep only plain text content and re-instruct.
                    messages.append(AIMessage(content=getattr(resp, "content", "") or ""))
                    messages.append(
                        HumanMessage(
                            content=(
                                f"Tool '{name}' does not exist. Available tools: {list(tool_map.keys())}. "
                                "Use exactly one valid tool call at a time, or return final JSON."
                            )
                        )
                    )
                    break

                if name == "get_matching_content":
                    query_raw = args.get("query", "") if isinstance(args, dict) else ""
                    q_norm = _normalize_tool_query(query_raw)
                    if q_norm and q_norm in used_queries:
                        logger.info(f"Round {round_num + 1}: Duplicate query skipped: {query_raw!r}")
                        duplicate_result = [
                            f"Duplicate query skipped: '{query_raw}'. Use a distinct keyword or return final JSON."
                        ]
                        messages.append(
                            ToolMessage(
                                content=sanitize_tool_result(duplicate_result, name),
                                tool_call_id=tool_id,
                            )
                        )
                        continue
                    if q_norm:
                        used_queries.add(q_norm)

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

                total_tool_calls += 1
                new_snippets = _collect_retrieval_snippets(name, result)
                if new_snippets:
                    retrieved_snippets.extend(new_snippets)
                    no_new_evidence_streak = 0
                elif name == "get_matching_content":
                    no_new_evidence_streak += 1
                    logger.info(
                        f"Round {round_num + 1}: query returned no new retrieval evidence "
                        f"(streak={no_new_evidence_streak})."
                    )

                messages.append(
                    ToolMessage(content=sanitize_tool_result(result, name), tool_call_id=tool_id)
                )

                if name == "get_matching_content" and no_new_evidence_streak >= 2:
                    logger.warning(
                        f"Round {round_num + 1}: {no_new_evidence_streak} consecutive retrieval calls "
                        "returned no new evidence. Forcing final JSON with cached evidence."
                    )
                    _push_cached_evidence_finalization(
                        "Recent retrieval queries returned no new evidence."
                    )
                    budget_exhausted_mid_round = True
                    break

            if total_tool_calls >= max_total_tool_calls:
                logger.warning(
                    f"Tool budget exhausted ({total_tool_calls}/{max_total_tool_calls}). "
                    "Next round will force final JSON using cached evidence."
                )
                _push_cached_evidence_finalization("Tool budget is exhausted.")

            if budget_exhausted_mid_round:
                continue
            continue

        content_raw = getattr(resp, "content", "") or ""
        # Strip LLaMA special tokens (e.g. <|python_tag|>) that bleed into content
        # when vLLM's tool parser fails. Keep the flag so we can detect the failure.
        has_python_tag = "<|python_tag|>" in content_raw
        content = re.sub(r"<\|[^|>]+\|>", "", content_raw).strip()
        last_content = content
        messages.append(AIMessage(content=content))

        # <|python_tag|> in content = vLLM tool parser ran but hit KeyError: 'name'
        if has_python_tag:
            logger.warning(
                f"Round {round_num + 1}: Content contains <|python_tag|> — "
                "tool call parse failed (model omitted 'name'). Asking model to retry."
            )
            messages.append(
                HumanMessage(
                    content=(
                        "Your tool call was malformed — the 'name' field is missing. Required format:\n"
                        '{"name": "<tool_name>", "parameters": {<args>}}\n'
                        f"Available tools: {list(tool_map.keys())}. Try again."
                    )
                )
            )
            continue

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
            parsed = _extract_json(content)

            # Detect text-embedded tool calls: finllama and some vLLM models output tool
            # calls as JSON in the message body instead of the structured tool_calls field.
            # Example: {"name": "get_matching_content", "parameters": {"query": "..."}}
            # If we returned this as the final result it would land in factor_extract as-is.
            # Catch nameless tool call that slipped past the json.loads check above
            # (e.g. {"parameters": {...}} with no "name"). Returning this as the
            # final answer would corrupt factor_extract.
            if isinstance(parsed, dict) and "parameters" in parsed and "name" not in parsed:
                logger.warning(
                    f"Round {round_num + 1}: _extract_json returned a dict with "
                    "'parameters' but no 'name' — treating as failed tool call."
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "Your response is a tool call missing the 'name' field. Required format:\n"
                            '{"name": "<tool_name>", "parameters": {<args>}}\n'
                            f"Available tools: {list(tool_map.keys())}. Try again."
                        )
                    )
                )
                continue
            if isinstance(parsed, dict):
                tc_name = parsed.get("name")
                tc_params = parsed.get("parameters") or parsed.get("args") or parsed.get("arguments")
                has_text_tool_shape = isinstance(tc_name, str) and (tc_params is not None)

                if has_text_tool_shape:
                    if tc_name in tool_map and tools_available:
                        if total_tool_calls >= max_total_tool_calls:
                            logger.warning(
                                f"Round {round_num + 1}: Tool budget exhausted ({total_tool_calls}/{max_total_tool_calls}) "
                                "before text-embedded tool call execution."
                            )
                            _push_cached_evidence_finalization("Tool budget is exhausted.")
                            continue

                        logger.info(
                            f"Round {round_num + 1}: Text-embedded tool call detected for '{tc_name}'. Executing."
                        )
                        tc_params_coerced = _coerce_tool_args(tc_params) if isinstance(tc_params, dict) else {}

                        if tc_name == "get_matching_content":
                            query_raw = tc_params_coerced.get("query", "") if isinstance(tc_params_coerced, dict) else ""
                            q_norm = _normalize_tool_query(query_raw)
                            if q_norm and q_norm in used_queries:
                                logger.info(f"Round {round_num + 1}: Duplicate text-embedded query skipped: {query_raw!r}")
                                messages.append(
                                    HumanMessage(
                                        content=(
                                            f"Duplicate query skipped: '{query_raw}'. "
                                            "Use a distinct query or return final JSON now."
                                        )
                                    )
                                )
                                continue
                            if q_norm:
                                used_queries.add(q_norm)

                        try:
                            tool_result = tool_map[tc_name].invoke(tc_params_coerced)
                        except Exception as e:
                            logger.warning(f"Tool '{tc_name}' raised {type(e).__name__}: {e}")
                            tool_result = f"Tool error: {type(e).__name__}: {str(e)[:200]}"

                        total_tool_calls += 1
                        new_snippets = _collect_retrieval_snippets(tc_name, tool_result)
                        if new_snippets:
                            retrieved_snippets.extend(new_snippets)
                            no_new_evidence_streak = 0
                        elif tc_name == "get_matching_content":
                            no_new_evidence_streak += 1
                            logger.info(
                                f"Round {round_num + 1}: text-embedded query returned no new "
                                f"retrieval evidence (streak={no_new_evidence_streak})."
                            )

                        messages.append(
                            HumanMessage(
                                content=(
                                    f"Tool result for '{tc_name}':\n"
                                    + sanitize_tool_result(tool_result, tc_name)
                                )
                            )
                        )
                        if total_tool_calls >= max_total_tool_calls:
                            logger.warning(
                                f"Tool budget exhausted ({total_tool_calls}/{max_total_tool_calls}). "
                                "Next round will force final JSON using cached evidence."
                            )
                            _push_cached_evidence_finalization("Tool budget is exhausted.")
                        if tc_name == "get_matching_content" and no_new_evidence_streak >= 2:
                            logger.warning(
                                f"Round {round_num + 1}: {no_new_evidence_streak} consecutive text-embedded "
                                "retrieval calls returned no new evidence. Forcing final JSON with cached evidence."
                            )
                            _push_cached_evidence_finalization(
                                "Recent retrieval queries returned no new evidence."
                            )
                        continue

                    if not tools_available:
                        logger.warning(
                            f"Round {round_num + 1}: Model returned a tool call after tool budget exhaustion. "
                            "Requesting final JSON without tools."
                        )
                        _push_cached_evidence_finalization("Tool budget is exhausted.")
                        continue

                    logger.warning(
                        f"Round {round_num + 1}: Text-embedded unknown tool '{tc_name}'. "
                        f"Available tools: {list(tool_map.keys())}"
                    )
                    messages.append(
                        HumanMessage(
                            content=(
                                f"Tool '{tc_name}' does not exist. Available tools: {list(tool_map.keys())}. "
                                "Use a valid tool or return final JSON."
                            )
                        )
                    )
                    continue

            return parsed
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
    evidence = _cached_evidence_blob(retrieved_snippets)
    evidence_text = evidence if evidence else "No cached evidence available."
    messages.append(
        HumanMessage(
            content=(
                "STOP. You have used all available rounds.\n"
                "Do not call tools again. Use ONLY the cached evidence below.\n\n"
                f"Cached evidence:\n{evidence_text}\n\n"
                "Return ONLY the final JSON object now. No tools. No prose."
            )
        )
    )
    try:
        resp = llm.invoke(messages)
    except Exception as e:
        logger.warning(
            f"Forced final JSON invoke failed ({type(e).__name__}: {e}). "
            "Returning empty JSON for caller-level fallback."
        )
        if last_content:
            try:
                return _extract_json(last_content)
            except ValueError:
                pass
        return {}
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
        return {"endpoint_valid": True, "endpoint_error": ""}

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
        error_msg = f"Cannot reach vLLM at {models_url}: {e}"
        logger.error(f"[validate_endpoint] {error_msg}")
        return {"endpoint_valid": False, "endpoint_error": error_msg}

    if model_name not in available:
        error_msg = f"Model '{model_name}' not served by vLLM. Available: {available}"
        logger.error(f"[validate_endpoint] {error_msg}")
        return {"endpoint_valid": False, "endpoint_error": error_msg}

    logger.info(f"Endpoint OK — '{model_name}' is available.")
    return {"endpoint_valid": True, "endpoint_error": ""}


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

_PLACEHOLDER_TOKENS = {"", "str", "string", "purpose", "computation_logic", "tbd", "<...>"}
_LOW_INFO_PATTERNS = {
    "factor replication",
    "cross-sectional regression",
    "cross sectional regression",
}
_META_PATTERNS = {
    "given paper snippet",
    "paper snippet",
    "use evidence from the snippet",
    "replicate the results",
    "replicate results",
    "apply the same methodology",
    "identify relevant factors",
    "extract factor replication details",
}
_META_STARTS = (
    "extract ",
    "use ",
    "identify ",
    "replicate ",
    "analyze ",
)
_DETAIL_CUES = (
    "rank",
    "sort",
    "decile",
    "quintile",
    "tercile",
    "long-short",
    "top-minus-bottom",
    "regression",
    "holding period",
    "rebalance",
    "monthly",
    "quarterly",
    "annual",
    "lag",
    "standardiz",
    "winsoriz",
    "value-weight",
    "equal-weight",
    "breakpoint",
    "filter",
    "exclude",
)
_MIN_PURPOSE_WORDS = 8
_MIN_COMPUTATION_WORDS = 24
_MIN_COMPUTATION_DETAIL_HITS = 3


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


_HTML_TAG_RE = re.compile(r"<(?:table|tr|td|th|div|span|p|br)[^>]*>", re.IGNORECASE)


def _is_low_information_field(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return True
    if value == "missing_or_unclear":
        return True
    if value in _LOW_INFO_PATTERNS:
        return True
    if any(p in value for p in _META_PATTERNS):
        return True
    if value.startswith(_META_STARTS) and ("snippet" in value or "paper" in value):
        return True
    # Raw HTML table markup pasted verbatim by the model is not useful prose
    if _HTML_TAG_RE.search(value):
        return True
    return _word_count(value) < 3


def _detail_cue_hits(text: str) -> int:
    value = (text or "").lower()
    hits = sum(1 for cue in _DETAIL_CUES if cue in value)
    if re.search(r"\b\d+(?:\.\d+)?(?:-\w+)?\b", value):
        hits += 1
    return hits


def _is_low_information_extract(payload: Dict[str, str]) -> bool:
    purpose = payload.get("purpose", "")
    computation_logic = payload.get("computation_logic", "")
    if _is_low_information_field(purpose) or _is_low_information_field(computation_logic):
        return True
    if _word_count(purpose) < _MIN_PURPOSE_WORDS:
        return True
    if _word_count(computation_logic) < _MIN_COMPUTATION_WORDS:
        return True
    if _detail_cue_hits(computation_logic) < _MIN_COMPUTATION_DETAIL_HITS:
        return True
    return False


def _factor_extract_quality_score(payload: Dict[str, str]) -> float:
    purpose = payload.get("purpose", "")
    computation_logic = payload.get("computation_logic", "")
    return (
        (0.5 * _word_count(purpose))
        + _word_count(computation_logic)
        + (2.5 * _detail_cue_hits(computation_logic))
    )


def _normalize_factor_extract(payload: Any) -> Dict[str, str]:
    """Guarantee non-empty, schema-safe factor_extract fields."""
    if not isinstance(payload, dict):
        payload = {}

    def _clean(value: Any) -> str:
        text = str(value or "").strip()
        # Strip HTML markup that small models sometimes copy verbatim from paper tables
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r" {2,}", " ", text).strip()
        if text.lower() in _PLACEHOLDER_TOKENS:
            return "missing_or_unclear"
        return text or "missing_or_unclear"

    return {
        "purpose": _clean(payload.get("purpose")),
        "computation_logic": _clean(payload.get("computation_logic")),
    }


def _rescue_alternative_theme_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Remap known alternative theme-pick schemas to the required one.

    finllama produces:  {id, description, investmentThesis, keyDrivers, investmentApproach, example}
    Required schema:    {theme, confidence, rationale, evidence_phrases}

    Only remaps fields that are missing or clearly wrong; existing valid fields
    are never overwritten.
    """
    if not isinstance(raw, dict):
        return raw

    rescued = dict(raw)

    # ---- theme rescue: map 'id' → 'theme' when theme is absent or invalid ----
    if rescued.get("theme") not in ALLOWED_THEMES:
        raw_id = str(rescued.get("id") or "").strip()
        matched = next(
            (t for t in ALLOWED_THEMES if t.lower() == raw_id.lower()),
            None,
        )
        if matched:
            rescued["theme"] = matched
            logger.info(f"[triage rescue] Mapped 'id'={raw_id!r} → theme={matched!r}")
            # Restore confidence to a non-zero default if it was zeroed out
            try:
                if float(rescued.get("confidence", 0)) == 0.0:
                    rescued["confidence"] = 0.7
            except (TypeError, ValueError):
                rescued["confidence"] = 0.7

    # ---- rationale rescue: pull from investmentThesis / investmentApproach / description ----
    if not (rescued.get("rationale") or "").strip():
        for src in ("investmentThesis", "investmentApproach", "description"):
            text = str(rescued.get(src) or "").strip()
            if text:
                rescued["rationale"] = text[:600]
                logger.info(f"[triage rescue] Mapped '{src}' → rationale")
                break

    # ---- evidence_phrases rescue: pull names from keyDrivers list ----
    if not (rescued.get("evidence_phrases") or []):
        drivers = rescued.get("keyDrivers") or []
        if isinstance(drivers, list):
            phrases = []
            for kd in drivers[:5]:
                if isinstance(kd, dict):
                    name = str(kd.get("name") or "").strip()
                    desc = str(kd.get("description") or "").strip()
                    phrase = f"{name}: {desc}"[:120] if (name and desc) else (name or desc)
                    if phrase:
                        phrases.append(phrase)
                elif isinstance(kd, str) and kd.strip():
                    phrases.append(kd.strip()[:120])
            if phrases:
                rescued["evidence_phrases"] = phrases
                logger.info(f"[triage rescue] Mapped 'keyDrivers' → {len(phrases)} evidence_phrases")

    return rescued


def triage_paper_node(state: AgentState) -> AgentState:
    """Agent 1: classify paper theme using the abstract."""
    paper_text = state.get("paper_text", "")
    abstract = _extract_abstract(paper_text)

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    prompt = build_system_prompt(AGENT1_PROMPT, max_tool_calls=0, abstract=abstract)

    def _parse_or_repair_theme_pick(raw_content: str) -> Dict[str, Any]:
        try:
            return _extract_json(raw_content)
        except Exception as parse_err:
            logger.warning(f"Paper triage JSON parse failed: {parse_err}. Trying repair.")
            repaired = _force_json(llm, raw_content)
            if isinstance(repaired, dict) and not repaired.get("_error"):
                return repaired
            raise

    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", "") or ""
        theme_pick = _parse_or_repair_theme_pick(raw)

    except Exception as first_err:
        logger.warning(
            f"Paper triage first attempt failed: {first_err}. Retrying once with stricter compact JSON instructions."
        )
        retry_prompt = (
            prompt
            + "\nIMPORTANT: Return compact JSON only.\n"
            + "Keep rationale <= 2 short sentences.\n"
            + "Each evidence phrase <= 12 words.\n"
            + "Return at most 3 evidence_phrases.\n"
        )
        try:
            resp2 = llm.invoke(retry_prompt)
            raw2 = getattr(resp2, "content", "") or ""
            theme_pick = _parse_or_repair_theme_pick(raw2)
        except Exception as e:
            logger.error(f"Paper triage failed: {e}")
            theme_pick = {
                "theme": "unknown",
                "confidence": 0.0,
                "rationale": str(e),
                "evidence_phrases": [],
            }

    try:
        theme_pick = _rescue_alternative_theme_schema(theme_pick)
        theme_pick.setdefault("confidence", 0.0)
        theme_pick["confidence"] = max(0.0, min(1.0, float(theme_pick["confidence"])))
        if theme_pick.get("theme") not in ALLOWED_THEMES:
            theme_pick["theme"] = "unknown"
            theme_pick["confidence"] = 0.0
        theme_pick.setdefault("rationale", "")
        theme_pick.setdefault("evidence_phrases", [])
        theme_pick["evidence_phrases"] = theme_pick["evidence_phrases"][:5]
    except Exception as e:
        logger.error(f"Paper triage normalization failed: {e}")
        theme_pick = {
            "theme": "unknown",
            "confidence": 0.0,
            "rationale": str(e),
            "evidence_phrases": [],
        }

    _log_agent_output("Agent 1 (triage)", theme_pick)

    triage_gcs_uri = ""
    storage_client = get_storage_client()
    if storage_client is not None:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            blob_name = (
                f"{state.get('paper_name', 'unknown')}/"
                f"{state.get('model_choice', 'unknown')}/"
                f"agent1_triage_{ts}.json"
            )
            triage_gcs_uri = upload_agent_output(storage_client, blob_name, theme_pick)
            logger.info(f"[Agent1] Triage checkpoint saved to {triage_gcs_uri}")
        except Exception as e:
            logger.warning(f"[Agent1] Triage GCS checkpoint failed: {e}")

    return {
        "abstract": abstract,
        "theme_pick": theme_pick,
        "triage_gcs_uri": triage_gcs_uri,
    }


def agent2_extract_factor_node(state: AgentState) -> AgentState:
    """Agent 2: extract factor purpose and computation logic."""
    paper_text = state.get("paper_text", "")
    _PAPER_TEXT.set(paper_text)
    _build_paper_index(paper_text)   # build BM25 index once before the ReAct loop

    model_choice = state.get("model_choice", "openai")
    llm = get_llm_by_name(model_choice)

    abstract = state.get("abstract", "")
    payload = {
        "theme_pick": state.get("theme_pick", {}),
        "abstract": abstract,
    }

    retrieved_snippets: List[str] = []
    used_queries: Set[str] = set()
    factor_extract = _normalize_factor_extract({})

    try:
        attempt_plans = (
            [(6, 6), (8, 7)]
            if model_choice == "custom"
            else [(8, 8), (12, 9)]
        )
        best_factor_extract = _normalize_factor_extract({})
        best_quality_score = -1.0

        for attempt_idx, (budget, rounds) in enumerate(attempt_plans, start=1):
            attempt_prompt = build_system_prompt(
                AGENT2_EXTRACT_PROMPT,
                max_tool_calls=budget,
                abstract=abstract,
            )
            if attempt_idx > 1:
                attempt_prompt += (
                    "\n\nRETRY REQUIREMENTS:\n"
                    "- Previous output was too generic.\n"
                    "- Include concrete implementation details in computation_logic, "
                    "including ranking/regression setup, timing/holding period, and portfolio construction when present.\n"
                    "- Do not summarize at a high level."
                )

            try:
                raw_factor_extract = _run_react_json_agent(
                    llm=llm,
                    system_prompt=attempt_prompt,
                    user_payload=payload,
                    tools=[get_matching_content],
                    max_rounds=rounds,
                    max_tool_calls_per_round=2,
                    max_total_tool_calls=budget,
                    retrieved_snippets=retrieved_snippets,
                    used_queries=used_queries,
                )
            except Exception as e:
                logger.warning(
                    f"Agent 2 attempt {attempt_idx}/{len(attempt_plans)} failed "
                    f"(budget={budget}, rounds={rounds}): {type(e).__name__}: {e}"
                )
                continue

            if not isinstance(raw_factor_extract, dict):
                raw_factor_extract = {}

            # Hard schema guard: prevent leaked tool-call payloads from being persisted.
            if any(k in raw_factor_extract for k in ("name", "parameters", "args", "arguments")):
                logger.warning("Agent 2 returned tool-call shaped JSON as final output. Coercing to empty schema.")
                raw_factor_extract = {}

            candidate = _normalize_factor_extract(raw_factor_extract)
            quality_score = _factor_extract_quality_score(candidate)
            if quality_score > best_quality_score:
                best_factor_extract = candidate
                best_quality_score = quality_score

            if not _is_low_information_extract(candidate):
                factor_extract = candidate
                break

            if attempt_idx < len(attempt_plans):
                logger.warning(
                    f"Agent 2 attempt {attempt_idx}/{len(attempt_plans)} was low-information "
                    f"(budget={budget}). Retrying with a larger budget."
                )
            else:
                logger.warning(
                    f"Agent 2 attempt {attempt_idx}/{len(attempt_plans)} was low-information "
                    f"(budget={budget}). Proceeding to fallback extraction."
                )
        else:
            factor_extract = best_factor_extract

        if retrieved_snippets:
            logger.info(
                f"Agent 2 cached {len(retrieved_snippets)} retrieval snippets "
                f"from {len(used_queries)} unique queries."
            )

        # Fallback path: if either field is still unclear, synthesize from cached evidence.
        if _is_low_information_extract(factor_extract):
            logger.warning("Agent 2 output incomplete or low-information. Running snippet-based fallback extraction.")
            fallback_extract = fallback_factor_extract_from_snippets(
                llm=llm,
                abstract=abstract,
                theme_pick=state.get("theme_pick", {}),
                query_fn=get_matching_content,
                json_parser=_extract_json,
                logger=logger,
                prefetched_snippets=retrieved_snippets,
                max_attempts=3,
            )
            fallback_extract = _normalize_factor_extract(fallback_extract)
            factor_extract = {
                "purpose": (
                    fallback_extract["purpose"]
                    if (
                        _is_low_information_field(factor_extract.get("purpose", ""))
                        or _word_count(factor_extract.get("purpose", "")) < _word_count(fallback_extract.get("purpose", ""))
                    )
                    else factor_extract.get("purpose", "missing_or_unclear")
                ),
                "computation_logic": (
                    fallback_extract["computation_logic"]
                    if (
                        _is_low_information_field(factor_extract.get("computation_logic", ""))
                        or _factor_extract_quality_score({
                            "purpose": factor_extract.get("purpose", ""),
                            "computation_logic": factor_extract.get("computation_logic", ""),
                        }) < _factor_extract_quality_score({
                            "purpose": fallback_extract.get("purpose", ""),
                            "computation_logic": fallback_extract.get("computation_logic", ""),
                        })
                    )
                    else factor_extract.get("computation_logic", "missing_or_unclear")
                ),
            }

        # Deterministic safety net: avoid all-missing outputs when LLM fallback times out.
        if _is_low_information_extract(factor_extract):
            heuristic_extract = heuristic_factor_extract_from_snippets(
                abstract=abstract,
                snippets=retrieved_snippets,
            )
            heuristic_extract = _normalize_factor_extract(heuristic_extract)
            if not _is_low_information_extract(heuristic_extract):
                logger.warning("Agent 2 using heuristic snippet fallback after low-information LLM output.")
                factor_extract = heuristic_extract
    except Exception as e:
        logger.error(f"Factor extraction failed: {e}")
        heuristic_extract = heuristic_factor_extract_from_snippets(
            abstract=abstract,
            snippets=retrieved_snippets,
        )
        factor_extract = _normalize_factor_extract(heuristic_extract)

    _log_agent_output("Agent 2 (factor_extract)", factor_extract)

    factor_gcs_uri = ""
    storage_client = get_storage_client()
    if storage_client is not None:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            blob_name = (
                f"{state.get('paper_name', 'unknown')}/"
                f"{state.get('model_choice', 'unknown')}/"
                f"agent2_factor_{ts}.json"
            )
            factor_gcs_uri = upload_agent_output(storage_client, blob_name, factor_extract)
            logger.info(f"[Agent2] Factor checkpoint saved to {factor_gcs_uri}")
        except Exception as e:
            logger.warning(f"[Agent2] Factor GCS checkpoint failed: {e}")

    return {
        "factor_extract": factor_extract,
        "factor_gcs_uri": factor_gcs_uri,
    }


def save_output_to_gcs_node(state: AgentState) -> AgentState:
    """Save final agent outputs to the GCS output bucket as JSON."""
    paper_name = state.get("paper_name", "unknown")
    model_choice = state.get("model_choice", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    blob_name = f"{paper_name}/{model_choice}/{timestamp}.json"

    normalized_factor_extract = _normalize_factor_extract(state.get("factor_extract", {}))
    payload = {
        "paper_name": paper_name,
        "model_choice": state.get("model_choice", ""),
        "theme_pick": state.get("theme_pick", {}),
        "factor_extract": normalized_factor_extract,
        "triage_gcs_uri": state.get("triage_gcs_uri", ""),
        "factor_gcs_uri": state.get("factor_gcs_uri", ""),
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
# Graph routing
# ---------------------------------------------------------------------
def _route_after_validation(state: AgentState) -> str:
    """Route after validate_endpoint_node.

    - Validation failed  → back to select_model (user picks a different model)
    - Validation passed + paper text in state → skip GCS load, go straight to triage
    - Validation passed + no paper text      → fetch paper from GCS first
    """
    if not state.get("endpoint_valid", True):
        return "retry"
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

    # select_model always proceeds to validation
    graph.add_edge("select_model", "validate_endpoint")

    # After validation: loop back to select_model on failure,
    # or continue to paper loading / triage on success
    graph.add_conditional_edges(
        "validate_endpoint",
        _route_after_validation,
        {
            "retry": "select_model",
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
        "triage_gcs_uri": "",
        "factor_gcs_uri": "",
        "output_gcs_uri": "",
        "endpoint_valid": False,
        "endpoint_error": "",
        "random_seed": None,
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
