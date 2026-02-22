"""Paper text chunking helpers for BM25 retrieval."""

from __future__ import annotations

import re
from typing import List


def chunk_academic_markdown(text: str) -> List[str]:
    """Split paper markdown into paragraph-level chunks.

    Equations and inline math are preserved in chunk text so downstream
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
        if len(full) > 80:
            chunks.append(full)

    return chunks


def normalize_for_bm25(text: str) -> str:
    """Normalize text for BM25 scoring without mutating source chunks."""
    normalized = re.sub(r"\$\$.*?\$\$", " EQ_TOKEN ", text, flags=re.DOTALL)
    normalized = re.sub(r"\$[^$\n]{1,200}\$", " EQ_TOKEN ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized