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
from Agents.prompts import (
    AGENT1_PROMPT,
    AGENT2_SYSTEM,
    AGENT3_SYSTEM,
    AGENT325_SYSTEM,
    AGENT4_SYSTEM,
    AGENT45_SYSTEM,
    AGENT5_SYSTEM,
    AGENT6_SYSTEM,
)


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
                # Handle nested properties format (LLaMA sometimes wraps args in 'properties')
                if "properties" in args and isinstance(args["properties"], dict):
                    args = args["properties"]
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


# Patch for LLaMA 3 single tool-call constraint (must run before graph is used)
from Agents.react_single_tool import patch_agent_module
patch_agent_module()

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