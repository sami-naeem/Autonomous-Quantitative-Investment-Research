"""Fallback extraction for Agent 2 when tool-calling output is empty."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def _compact_snippets(snippets: List[str], max_items: int = 8, max_chars: int = 6000) -> str:
    unique: List[str] = []
    seen = set()
    total = 0
    for s in snippets:
        text = (s or "").strip()
        if not text or text in seen:
            continue
        if total + len(text) > max_chars:
            break
        unique.append(text)
        seen.add(text)
        total += len(text)
        if len(unique) >= max_items:
            break
    return "\n\n---\n\n".join(unique)


def fallback_factor_extract_from_snippets(
    *,
    llm: Any,
    abstract: str,
    theme_pick: Dict[str, Any],
    query_fn: Callable[[str], List[str]],
    json_parser: Callable[[str], Dict[str, Any]],
    logger: Any,
) -> Dict[str, str]:
    """Run one JSON-only extraction pass over deterministic BM25 snippets."""
    queries = [
        "factor definition",
        "portfolio construction",
        "sorting",
        "holding period",
        "rebalance",
        "long short",
    ]

    collected: List[str] = []
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
        return {"purpose": "", "computation_logic": ""}

    prompt = (
        "You are extracting factor replication details from paper snippets.\n"
        "Return ONLY a valid JSON object with exactly these keys:\n"
        '{"purpose": "str", "computation_logic": "str"}\n'
        "Use only evidence in snippets and abstract. If uncertain, be concise and state assumptions.\n\n"
        f"Theme pick:\n{theme_pick}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Snippets:\n{snippets_blob}\n"
    )

    try:
        resp = llm.invoke(prompt)
        parsed = json_parser(getattr(resp, "content", "") or "")
        if not isinstance(parsed, dict):
            return {"purpose": "", "computation_logic": ""}
        return {
            "purpose": str(parsed.get("purpose", "") or ""),
            "computation_logic": str(parsed.get("computation_logic", "") or ""),
        }
    except Exception as e:
        logger.warning(f"Fallback extraction call failed: {type(e).__name__}: {e}")
        return {"purpose": "", "computation_logic": ""}
