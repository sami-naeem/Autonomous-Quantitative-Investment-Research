"""Fallback extraction for Agent 2 when tool-calling output is empty."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

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


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def _is_low_information_field(text: str) -> bool:
    value = (text or "").strip().lower()
    if not value:
        return True
    if value in _LOW_INFO_PATTERNS:
        return True
    if any(p in value for p in _META_PATTERNS):
        return True
    if value.startswith(_META_STARTS) and ("snippet" in value or "paper" in value):
        return True
    return _word_count(value) < 3


def _is_low_information_extract(payload: Dict[str, str]) -> bool:
    return (
        _is_low_information_field(payload.get("purpose", ""))
        or _is_low_information_field(payload.get("computation_logic", ""))
    )


def _snippet_score(snippet: str) -> float:
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


def _compact_snippets(snippets: List[str], max_items: int = 8, max_chars: int = 6000) -> str:
    best_by_text: Dict[str, float] = {}
    for s in snippets:
        text = (s or "").strip()
        if not text:
            continue
        score = _snippet_score(text)
        prev = best_by_text.get(text)
        if prev is None or score > prev:
            best_by_text[text] = score

    ranked = sorted(best_by_text.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[str] = []
    total = 0
    for text, _ in ranked:
        if total + len(text) > max_chars:
            break
        selected.append(text)
        total += len(text)
        if len(selected) >= max_items:
            break
    return "\n\n---\n\n".join(selected)


def _normalize_factor_extract(payload: Any) -> Dict[str, str]:
    if not isinstance(payload, dict):
        payload = {}

    def _clean(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower() in _PLACEHOLDER_TOKENS:
            return "missing_or_unclear"
        return text or "missing_or_unclear"

    return {
        "purpose": _clean(payload.get("purpose")),
        "computation_logic": _clean(payload.get("computation_logic")),
    }


def fallback_factor_extract_from_snippets(
    *,
    llm: Any,
    abstract: str,
    theme_pick: Dict[str, Any],
    query_fn: Callable[[str], List[str]],
    json_parser: Callable[[str], Dict[str, Any]],
    logger: Any,
    prefetched_snippets: List[str] | None = None,
) -> Dict[str, str]:
    """Run one JSON-only extraction pass over cached/retrieved BM25 snippets."""
    queries = [
        "factor definition",
        "portfolio construction",
        "sorting",
        "holding period",
        "rebalance",
        "long short",
    ]

    collected: List[str] = list(prefetched_snippets or [])
    if not collected:
        for q in queries:
            try:
                if hasattr(query_fn, "invoke"):
                    # LangChain StructuredTool path
                    res = query_fn.invoke({"query": q})
                else:
                    # Plain callable path
                    res = query_fn(q)
                collected.extend(res or [])
            except Exception as e:
                logger.warning(f"Fallback snippet query failed for {q!r}: {type(e).__name__}: {e}")

    snippets_blob = _compact_snippets(collected)
    if not snippets_blob:
        return _normalize_factor_extract({})

    prompt = (
        "You are extracting factor replication details from paper snippets.\n"
        "Return ONLY a valid JSON object with exactly these keys:\n"
        '{"purpose": "str", "computation_logic": "str"}\n'
        "Never output literal placeholders. Do not return values like 'str', 'string', '<...>', or 'TBD'.\n"
        "Use only evidence in snippets and abstract. If uncertain, be concise and state assumptions.\n\n"
        f"Theme pick:\n{theme_pick}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Snippets:\n{snippets_blob}\n"
    )

    try:
        resp = llm.invoke(prompt)
        parsed = json_parser(getattr(resp, "content", "") or "")
        first_pass = _normalize_factor_extract(parsed)
        if not _is_low_information_extract(first_pass):
            return first_pass

        logger.warning(
            "Fallback extraction produced low-information output. Retrying once with stricter requirements."
        )
        strict_prompt = (
            "Your previous extraction was too generic.\n"
            "Retry and return ONLY valid JSON with keys purpose and computation_logic.\n"
            "Each field must be at least 3 words.\n"
            "Do not use generic responses like 'factor replication' or 'cross-sectional regression'.\n"
            "Do not write meta-instructions (e.g., 'use evidence from snippet', 'apply the same methodology').\n"
            "Use concrete details from snippets (e.g., variable definition, sorting/regression setup, timing).\n\n"
            f"Previous output:\n{first_pass}\n\n"
            f"Theme pick:\n{theme_pick}\n\n"
            f"Abstract:\n{abstract}\n\n"
            f"Snippets:\n{snippets_blob}\n"
        )
        resp2 = llm.invoke(strict_prompt)
        parsed2 = json_parser(getattr(resp2, "content", "") or "")
        return _normalize_factor_extract(parsed2)
    except Exception as e:
        logger.warning(f"Fallback extraction call failed: {type(e).__name__}: {e}")
        return _normalize_factor_extract({})
