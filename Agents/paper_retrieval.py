"""Paper text chunking helpers for BM25 retrieval."""

from __future__ import annotations

import re
from typing import List


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text, collapsing runs of whitespace left behind."""
    stripped = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r" {2,}", " ", stripped).strip()


def chunk_academic_markdown(text: str) -> List[str]:
    """Split paper markdown into paragraph-level chunks.

    HTML table markup (common in PDF-to-markdown output) is stripped before
    storing chunks so BM25 snippets returned to the model contain plain text
    only.  Equations and inline math are preserved in chunk text so downstream
    replication logic can inspect original formulas.
    """
    raw_chunks = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    header_buffer = ""

    for raw in raw_chunks:
        chunk = raw.strip()
        if not chunk:
            continue
        if re.match(r"^#{1,4}\s", chunk) and len(chunk) < 120:
            header_buffer = chunk + "\n"
            continue
        full = header_buffer + chunk
        header_buffer = ""
        # Strip HTML markup that PDF converters embed as tables/spans
        full = _strip_html_tags(full)
        if len(full) > 80:
            chunks.append(full)

    return chunks


def normalize_for_bm25(text: str) -> str:
    """Normalize text for BM25 scoring without mutating source chunks."""
    normalized = re.sub(r"\$\$.*?\$\$", " EQ_TOKEN ", text, flags=re.DOTALL)
    normalized = re.sub(r"\$[^$\n]{1,200}\$", " EQ_TOKEN ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized