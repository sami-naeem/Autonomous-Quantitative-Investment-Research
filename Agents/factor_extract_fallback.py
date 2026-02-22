"""Fallback extraction for Agent 2 when tool-calling output is empty."""

from __future__ import annotations

import re
import time
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
    return (
        _is_low_information_field(purpose)
        or _is_low_information_field(computation_logic)
        or _word_count(purpose) < _MIN_PURPOSE_WORDS
        or _word_count(computation_logic) < _MIN_COMPUTATION_WORDS
        or _detail_cue_hits(computation_logic) < _MIN_COMPUTATION_DETAIL_HITS
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


def _compact_snippets(snippets: List[str], max_items: int = 6, max_chars: int = 3500) -> str:
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
    max_attempts: int = 3,
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

    best = _normalize_factor_extract({})
    best_score = -1

    for attempt_idx in range(max_attempts):
        attempt_no = attempt_idx + 1
        if attempt_idx == 0:
            active_prompt = prompt
        else:
            active_prompt = (
                "Your previous extraction was too generic.\n"
                "Retry and return ONLY valid JSON with keys purpose and computation_logic.\n"
                f"Purpose must be at least {_MIN_PURPOSE_WORDS} words.\n"
                f"Computation logic must be at least {_MIN_COMPUTATION_WORDS} words.\n"
                "Do not use generic responses like 'factor replication' or 'cross-sectional regression'.\n"
                "Do not write meta-instructions (e.g., 'use evidence from snippet', 'apply the same methodology').\n"
                "Use concrete details from snippets (e.g., variable definition, sorting/regression setup, timing).\n"
                "Include at least three concrete implementation cues in computation_logic.\n\n"
                f"Previous output:\n{best}\n\n"
                f"Theme pick:\n{theme_pick}\n\n"
                f"Abstract:\n{abstract}\n\n"
                f"Snippets:\n{snippets_blob}\n"
            )

        logger.info(f"Fallback extraction attempt {attempt_no}/{max_attempts} starting.")
        t0 = time.perf_counter()
        try:
            resp = llm.invoke(active_prompt)
            parsed = json_parser(getattr(resp, "content", "") or "")
            candidate = _normalize_factor_extract(parsed)
        except Exception as e:
            logger.warning(
                f"Fallback extraction attempt {attempt_no}/{max_attempts} failed after "
                f"{time.perf_counter() - t0:.2f}s: {type(e).__name__}: {e}"
            )
            continue

        elapsed = time.perf_counter() - t0
        logger.info(f"Fallback extraction attempt {attempt_no}/{max_attempts} completed in {elapsed:.2f}s.")

        candidate_score = _word_count(candidate.get("purpose", "")) + _word_count(candidate.get("computation_logic", ""))
        if candidate_score > best_score:
            best = candidate
            best_score = candidate_score

        if not _is_low_information_extract(candidate):
            return candidate

        logger.warning(
            f"Fallback extraction attempt {attempt_no}/{max_attempts} produced low-information output."
        )

    return best


def heuristic_factor_extract_from_snippets(
    *,
    abstract: str,
    snippets: List[str],
) -> Dict[str, str]:
    """Deterministic extraction fallback that does not call an LLM."""
    cleaned_snippets: List[str] = []
    for raw in snippets or []:
        text = re.sub(r"^\[(?:MATCH[^\]]*|CONTEXT)\]\s*", "", str(raw or "").strip())
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text.startswith("No relevant content found"):
            continue
        cleaned_snippets.append(text)

    def _pick_sentences(text: str, max_sentences: int = 2) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text or "")
        out = []
        for p in parts:
            p = p.strip()
            if _word_count(p) < 6:
                continue
            out.append(p)
            if len(out) >= max_sentences:
                break
        return out

    abstract_sentences = _pick_sentences(abstract, max_sentences=2)
    purpose = " ".join(abstract_sentences).strip()

    if _word_count(purpose) < _MIN_PURPOSE_WORDS:
        for s in cleaned_snippets:
            snippet_sentences = _pick_sentences(s, max_sentences=2)
            candidate = " ".join(snippet_sentences).strip()
            if _word_count(candidate) >= _MIN_PURPOSE_WORDS:
                purpose = candidate
                break

    detail_candidates: List[tuple[int, str]] = []
    for s in cleaned_snippets:
        score = _detail_cue_hits(s)
        if score <= 0:
            continue
        detail_candidates.append((score, s))

    detail_candidates.sort(key=lambda x: x[0], reverse=True)
    comp_parts: List[str] = []
    for _, text in detail_candidates[:3]:
        for sentence in _pick_sentences(text, max_sentences=2):
            comp_parts.append(sentence)
            if len(comp_parts) >= 4:
                break
        if len(comp_parts) >= 4:
            break

    if not comp_parts and cleaned_snippets:
        comp_parts = _pick_sentences(" ".join(cleaned_snippets[:2]), max_sentences=3)

    computation_logic = " ".join(comp_parts).strip()
    result = _normalize_factor_extract(
        {
            "purpose": purpose,
            "computation_logic": computation_logic,
        }
    )
    return result
