# LLM Pipeline Hardening — Summary of All Changes & Improvements

## Overview

Complete rewrite of the ReAct-style multi-agent paper replication pipeline (`agent.py` → `agent_v2.py`, `prompts.py` → `prompts_v2.py`) to enforce strict JSON output from LLaMA 3.1 8B running on vLLM. Identified and fixed 12 root causes across three live test runs.

**Files delivered:** `agent_v2.py`, `prompts_v2.py`, `deliverable.md`

---

## Root Causes Identified (12 total)

| # | Root Cause | Severity |
|---|-----------|----------|
| 1 | No explicit stop-vs-tool-call contract — model loops 20+ tool calls | Critical |
| 2 | JSON schemas embedded with comments (`# each step one sentence`) primes model to output invalid JSON | High |
| 3 | No snippet length limit — tool results return multi-KB text, bloating context | High |
| 4 | `_extract_json` regex `\{.*\}` with `re.S` is greedy — grabs first `{` to last `}` across multiple objects | Critical |
| 5 | `_force_json` re-sends entire system prompt (~1500 tokens) as repair context | Medium |
| 6 | No sanitization of `ToolMessage` content — raw stack traces and Unicode junk leak into message history | High |
| 7 | `max_rounds=12/14` too generous — encourages model wandering | Medium |
| 8 | No "search plan" instruction — model invents queries indefinitely | Medium |
| 9 | Repair path exits on first non-JSON answer instead of allowing one retry | Medium |
| 10 | `ast.literal_eval` fallback is dangerous — can execute Python expressions | High |
| 11 | Agent 1 echoes "MODE A — TOOL CALL" as prose (found in live run 1) | Critical |
| 12 | vLLM/LLaMA 3.1 serializes `List[str]` tool args as JSON strings (found in live run 1) | Critical |

---

## Changes to `agent_v2.py`

### 1. `_extract_json()` — Rewritten

- **Before:** Greedy regex `\{.*\}` with `re.S`, plus unsafe `ast.literal_eval` fallback.
- **After:** Balanced-brace scanner that counts `{` / `}` depth to find the first complete JSON object. `ast.literal_eval` removed entirely. Added safe repair steps for trailing commas, JS-style comments, single quotes → double quotes, and `NaN`/`Infinity` → `null`.

### 2. `sanitize_tool_result()` — New Function

Cleans and truncates all tool output before it enters the message history:

- List results capped to 5 items (`MAX_SNIPPETS_PER_CALL`).
- Each string truncated to 600 characters (`MAX_SNIPPET_CHARS`).
- Stack traces stripped via regex.
- Log-line patterns removed.
- Always returns valid JSON string; exceptions produce a safe fallback payload.

### 3. `_force_json()` — Simplified

- **Before:** Re-sent the entire system prompt as schema instructions (~1500 tokens).
- **After:** Minimal 3-line repair prompt ("You previously produced text that is not valid JSON. Convert it to a single valid JSON object..."). Input truncated to 3000 characters to avoid blowing the context window.

### 4. `_coerce_tool_args()` — New Function

Workaround for vLLM/LLaMA 3.1 serialization bug where `List[str]` parameters arrive as JSON strings (e.g., `keys='["operating cash flow", "revenue"]'` instead of `keys=["operating cash flow", "revenue"]`). Detects string values that look like JSON arrays/objects and parses them back into native Python types before tool invocation.

### 5. `_run_react_json_agent()` — Hardened Loop

- **Tool budget enforcement:** Per-agent `max_total_tool_calls` parameter. Once budget is exhausted, the loop invokes `llm` (no tools bound) instead of `llm_with_tools`, making it physically impossible for the model to call more tools.
- **Per-round tool cap:** `max_tool_calls_per_round` (default 3) prevents batched over-calling.
- **One JSON retry before repair:** On first non-JSON final output, the model gets one retry with its accumulated context. Only on the second failure does it fall through to `_force_json`.
- **Malformed tool-call detection:** If the model emits JSON content that looks like a tool call (has `"parameters"` key, missing or matching a real tool `"name"`), it's caught and the model is asked to retry.
- **Truncated output handling:** Detects `finish_reason='length'` and asks the model to retry concisely.
- **Invalid tool-call handling:** Catches `invalid_tool_calls` from LangChain and prompts retry.
- **Unknown tool handling:** If the model hallucinates a tool name (e.g., `get_final_answer`), it's caught, logged, and the model is redirected.
- **Tool args unwrapping:** LLaMA sometimes wraps args in a `"properties"` key — automatically unwrapped.
- **One-tool-per-AIMessage:** LLaMA 3 / vLLM constraint enforced by splitting batch tool calls into individual `AIMessage` objects.

### 6. Agent Node Updates

All agent nodes now use `build_system_prompt()` with per-agent tool budgets:

| Agent | Tool Budget | Max Rounds | v1 Values |
|-------|------------|------------|-----------|
| Agent 1 (triage) | 0 (no tools) | 1 call | N/A |
| Agent 2 (paper_vars) | 6 | 8 | 12 rounds, unbounded |
| Agent 3 (paper_logic) | 8 | 10 | 14 rounds, unbounded |
| Agent 3.25 (paper_spec) | 0 (no tools) | 1 call + repair | N/A |
| Agent 4 (sharadar_map) | 10 | 12 | 16 rounds, unbounded |

### 7. `_log_agent_output()` — New Helper (Run 3)

Added terminal logging of each agent's JSON output (A1, A2, A3, A3.25, A4) with 3000-character truncation. Provides immediate visibility into agent outputs without requiring LangSmith Studio navigation.

---

## Changes to `prompts_v2.py`

### 1. Split JSON Contract

- **Before:** Single `JSON_CONTRACT` injected into all agents, containing both tool-call protocol ("MODE A — TOOL CALL") and final-JSON instructions ("MODE B — FINAL JSON").
- **After:** Split into two variants:
  - `_JSON_CONTRACT_BASE` — For tool-free agents (Agent 1, 3.25). Contains only final-JSON instructions. No "MODE A / MODE B" labels that LLaMA parrots back.
  - `_JSON_CONTRACT_TOOLS` — Extends base with tool-call protocol, tool budget countdown, and explicit "tools will be disabled" warning.

### 2. `build_system_prompt()` Function

New prompt builder that:

- Selects the correct JSON contract variant based on `max_tool_calls` parameter.
- Injects tool budget number into the prompt (e.g., "You have 6 tool calls remaining").
- Supports `**kwargs` for template substitution (e.g., `abstract=` for Agent 1).

### 3. Agent 2 Prompt — Rewritten with Search Plan

- Added explicit 3-step search plan: (1) search for key terms from abstract, (2) search for data sources / sample construction, (3) search for variable definitions.
- Explicit instruction: "You have exactly ONE tool: `get_matching_content`."
- Schema stripped of comments and pseudo-JSON examples.

### 4. Agent 3 Prompt — Rewritten with Search Plan

- Added 4-step search plan for factor logic extraction: sorting, portfolio construction, rebalancing, return calculation.
- Same "one tool only" reinforcement.

### 5. Agent 4 Prompt — Rewritten with Search Plan

- Added 3-step Sharadar mapping plan: (1) search fundamental fields, (2) search price/daily fields, (3) map paper quantities to best Sharadar matches.
- Explicit: "You have exactly ONE tool: `get_sh_matching_variables`."

---

## Live Test Results

### Run 1 — Two Critical Bugs Found

| Bug | Description | Fix |
|-----|------------|-----|
| #11 | Agent 1 echoed "MODE A — TOOL CALL" as literal text | Split JSON contract into base/tools variants |
| #12 | Agent 4 `ValidationError` on every tool call — `keys` parameter arrived as string | Added `_coerce_tool_args()` |

### Run 2 — Both Fixes Confirmed Working

- Pipeline completed end-to-end (~177s).
- Agent 4's `_coerce_tool_args` eliminated 100% of `ValidationError` failures.
- Split contract eliminated Agent 1's "MODE A" echo.
- Three remaining issues identified: duplicate execution, hallucinated tool names, Agent 3.25 needing repair.

### Run 3 — Output Logging Added, Full Visibility

- Pipeline completed in ~119s.
- All tool budgets enforced correctly (6/6, 8/8, 10/10).
- Per-agent output logging confirmed working.
- Content quality limited by test paper being wrong type (Heston volatility model, not a factor paper).

---

## Known Remaining Issues

| Issue | Description | Proposed Fix |
|-------|------------|-------------|
| Duplicate execution | All downstream nodes (A2–A4) run twice due to LangGraph fan-in from `load_paper_markdown` + `validate_endpoint` both triggering `triage_paper` | Restructure graph so triage waits for all predecessors before firing once |
| Agent 4 tool-call-as-content | When tools are disabled, Agent 4 emits tool-call JSON schema as text content (has `"name"` matching real tool name) | Extend detection: `maybe_tc.get("name") in tool_map` |
| Wrong test paper | Auto-selected paper `1404.4014v3` is a derivatives paper, not a cross-sectional factor paper | Add post-triage gate to skip non-factor papers, or specify `paper_name` in input |
| Agent hallucinating tool names | LLaMA invents `get_final_answer`, `get_quantities_and_universe`, `get_definitions` | Already caught and handled; could add explicit "no other tools exist" to prompts |

---

## Infrastructure Improvements Summary

| Capability | Before (v1) | After (v2) |
|-----------|-------------|------------|
| JSON extraction | Greedy regex + `ast.literal_eval` | Balanced-brace scanner, no eval |
| Tool result handling | Raw, unbounded | Sanitized, truncated, capped |
| Tool budget | Unbounded (20+ calls observed) | Per-agent budgets (6/8/10) |
| JSON repair | Re-sends full system prompt | Minimal 3-line repair prompt |
| JSON retry | Immediate repair on first failure | One in-loop retry, then repair |
| Tool arg types | Passed through as-is | `_coerce_tool_args` fixes vLLM serialization |
| Prompt structure | Single JSON contract for all agents | Split base/tools variants |
| Agent visibility | LangSmith only | Terminal output logging |
| Search guidance | "Search using different keywords" | Explicit multi-step search plans |