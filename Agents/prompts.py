"""
Agent system prompts for the paper replication pipeline (v2 — strict JSON contract).

Changes from v1:
- Shared JSON_CONTRACT injected into every agent prompt.
- Explicit search plans with tool-call budgets.
- Schemas shown as valid JSON (no comments, no trailing commas).
- Mini-examples for Agent 2 and Agent 3 to reduce invalid output.
- build_system_prompt() helper for assembly.
"""

# ---------------------------------------------------------------------------
# Shared JSON Contract — injected into every agent system prompt
# ---------------------------------------------------------------------------

# Base contract: for agents that do NOT use tools (Agent 1, 3.25, 4.5, 5, 6)
_JSON_CONTRACT_BASE = """=== JSON OUTPUT CONTRACT ===
1. Your ENTIRE response MUST be a single JSON object. Nothing else.
2. No markdown. No code fences. No prose before or after the JSON.
3. No comments inside JSON. No trailing commas. Use double quotes only.
4. Every string value must be ≤600 characters. Truncate with "..." if needed.
5. If information is not found, use the value "missing_or_unclear" (string) or
   add the item to the "missing_or_unclear" array. Never leave a required field empty.
6. evidence/snippets: each snippet ≤600 chars, plain text only, no LaTeX, no tables.
7. Never output literal placeholder tokens such as "str", "string", "<...>", "TBD", or field names as values.
"""

# Extended contract: for agents that DO use tools (Agent 2, 3, 4)
_JSON_CONTRACT_TOOLS = _JSON_CONTRACT_BASE + """
=== TOOL USE PROTOCOL ===
Each response must be EXACTLY ONE of these two modes:
  (A) A tool call — no text/JSON in the message body.
  (B) Your final JSON object — no tool calls.
Never combine a tool call and JSON output in the same response.

=== WHEN TO STOP CALLING TOOLS ===
Return your final JSON when ANY of these is true:
  - You have called tools {max_tool_calls} times total.
  - Every required schema field can be filled (even with "missing_or_unclear").
  - Your last 2 tool calls returned no new useful information.
Do NOT keep searching for perfection. Fill unknowns with "missing_or_unclear" and finish.
"""


def build_system_prompt(
    agent_prompt: str,
    max_tool_calls: int = 0,
    abstract: str = "",
) -> str:
    """Assemble final system prompt with the appropriate JSON contract injected."""
    if max_tool_calls > 0:
        filled_contract = _JSON_CONTRACT_TOOLS.format(max_tool_calls=max_tool_calls)
    else:
        filled_contract = _JSON_CONTRACT_BASE
    result = agent_prompt.replace("{json_contract}", filled_contract)
    if "{abstract}" in result:
        result = result.replace("{abstract}", abstract)
    return result


# ---------------------------------------------------------------------------
# Agent 1 — Paper Triage
# ---------------------------------------------------------------------------
AGENT1_PROMPT = """{json_contract}

ROLE: Classify the paper's primary investment theme based ONLY on the abstract below.

ALLOWED THEMES: Profitability, Low Leverage, Accruals, Investment, Debt Issuance, Size, Quality, Seasonality, Value, Momentum

SCHEMA (return EXACTLY this structure):
{{
  "theme": "str (one of the allowed themes above)",
  "confidence": 0.0-1.0,
  "rationale": "1-3 sentences explaining why this theme fits the abstract",
  "evidence_phrases": ["short phrases copied from the abstract that support the choice, max 5"]
}}

ABSTRACT:
{abstract}
"""

AGENT2_EXTRACT_PROMPT = """{json_contract}

ROLE: Extract the factor's purpose and computation logic from the paper.

TOOL: get_matching_content(query) — returns relevant paper paragraphs for a short keyword query.
Use 1-3 word queries (e.g. "portfolio sort", "holding period", "book-to-market").
Do NOT use long sentences — the paper may phrase things differently.

SEARCH PLAN (follow this order, then STOP):
1. Search 2-3 short keywords for factor definition/construction (e.g. "defined as", "ratio", "sort", "portfolio", "decile").
2. Search 1-2 short keywords for timing/rebalancing (e.g. "lag", "rebalance", "holding period").
3. You are done. Return final JSON. Do NOT exceed 5 total tool calls.

OUTPUT RULES:
- Return ONLY valid JSON with exactly two keys.
- "purpose": 1-2 sentences on what the factor measures.
- "computation_logic": prose describing how to compute the factor, detailed enough to implement.
- Never output literal placeholders. Do not use values like "str", "string", "<...>", "TBD", or key names.

STYLE RULES for computation_logic:
- No equations, no LaTeX, no math symbols.
- No bullet lists or numbered lists.
- Use short paragraphs separated by blank lines, each paragraph 4 sentences or fewer.
- Avoid dataset-specific mnemonics; use plain English meanings instead.
- Include timing/lagging, sorting/breakpoints, filters/exclusions, standardization, industry adjustment, holding period, and output frequency ONLY if the paper text supports them.

SCHEMA (return EXACTLY this structure):
{{
  "purpose": "str",
  "computation_logic": "str"
}}

ABSTRACT:
{abstract}
"""
