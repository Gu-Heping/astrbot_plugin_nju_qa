"""Convert grep hits and read documents into a unified evidence model."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .models import ChunkResult, Document, SearchResult

_QA_STATUSES = (
    "resolved",
    "reviewed_resolved",
    "partially_resolved",
    "no_answer_found",
)


def _normalize(text: str) -> str:
    """Case-insensitive comparison helper."""
    return text.casefold()


def _extract_terms(query: str) -> list[str]:
    """Return non-empty whitespace-separated terms."""
    return [t for t in query.split() if t]


def _combine_grep_snippets(hit: dict) -> str:
    """Merge grep match windows into one snippet body."""
    snippets = [
        match.get("snippet", "")
        for match in hit.get("matches", [])
        if match.get("snippet")
    ]
    return "\n\n".join(snippets)


def _window_texts(hit: dict) -> list[str]:
    """Return the text of every matched window."""
    return [
        match.get("snippet", "")
        for match in hit.get("matches", [])
        if match.get("snippet")
    ]


def _window_status(text: str) -> str | None:
    """Detect QA status markers inside a snippet window."""
    lower = _normalize(text)
    for status in _QA_STATUSES:
        if status in lower:
            return status
    return None


def _is_index_document(title: str) -> bool:
    """Heuristic for directory / table-of-contents documents."""
    lower = _normalize(title)
    if re.search(r"(^|[/_-])index\b|\\b目录\\b|\\btoc\\b|^00[_-]", lower):
        return True
    return lower.startswith("00_")


def build_document_from_grep_hit(hit: dict) -> Document:
    """Create a Document from a grep tool hit dict."""
    return Document(
        yuque_id=hit.get("yuque_id") or hit.get("id", ""),
        title=hit.get("title", ""),
        repository=hit.get("repository", ""),
        namespace=hit.get("namespace", ""),
        slug=hit.get("slug", ""),
        url=hit.get("url", ""),
        created_at=hit.get("created_at", ""),
        updated_at=hit.get("updated_at", ""),
        body=_combine_grep_snippets(hit),
        path=Path(hit["path"]) if hit.get("path") else None,
    )


def document_from_index_row(row, body: str = "") -> Document:
    """Create a Document from a DocumentIndex SQLite row."""
    return Document(
        yuque_id=row["yuque_id"],
        title=row["title"],
        repository=row["repository"],
        namespace=row["namespace"],
        slug=row["slug"],
        url=row["url"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        body=body or row["body"] or "",
        path=Path(row["path"]) if row["path"] else None,
    )


def evaluate_grep_reliability(hit: dict, query_terms: list[str]) -> tuple[bool, dict]:
    """Return (reliable, diagnostics) for a grep hit.

    Reliability requires real content and meaningful keyword coverage.  It does
    not blindly trust every line that mentions one of the query words.
    """
    terms = [_normalize(t) for t in query_terms if t]
    if not terms:
        return False, {"reason": "no_terms"}

    matched = {_normalize(t) for t in hit.get("matched_keywords", [])}
    coverage = len(matched) / len(terms)
    windows = _window_texts(hit)
    if not windows:
        return False, {"coverage": coverage, "reason": "no_windows"}

    same_window_full = any(
        all(term in _normalize(window) for term in terms) for window in windows
    )
    phrase = " ".join(query_terms)
    exact_phrase_match = any(phrase in window for window in windows)

    title = _normalize(hit.get("title", ""))
    title_all_terms = all(term in title for term in terms)
    title_any_term = any(term in title for term in terms)
    is_index = _is_index_document(hit.get("title", ""))

    statuses = [_window_status(window) for window in windows]
    has_resolved = any(
        status in ("resolved", "reviewed_resolved") for status in statuses
    )
    all_no_answer = statuses and all(
        status == "no_answer_found" for status in statuses
    )

    # Base rule: full coverage, real content, not a directory page.
    reliable = coverage >= 1.0 and not is_index

    # QA status overrides: a clearly resolved window is good;
    # a window that only says "no_answer_found" is not evidence.
    if all_no_answer:
        reliable = False
    elif has_resolved and same_window_full:
        reliable = True
    elif same_window_full or exact_phrase_match or title_all_terms:
        reliable = True
    else:
        reliable = False

    diagnostics = {
        "coverage": coverage,
        "same_window_full_coverage": same_window_full,
        "exact_phrase_match": exact_phrase_match,
        "title_all_terms": title_all_terms,
        "title_any_term": title_any_term,
        "is_index_document": is_index,
        "has_resolved": has_resolved,
        "all_no_answer_found": all_no_answer,
    }
    return reliable, diagnostics


def score_grep_hit(hit: dict, query_terms: list[str]) -> float:
    """Rank grep hits so direct answers outrank partial or index matches."""
    terms = [_normalize(t) for t in query_terms if t]
    if not terms:
        return 0.0

    matched = {_normalize(t) for t in hit.get("matched_keywords", [])}
    coverage = len(matched) / len(terms)
    windows = _window_texts(hit)
    same_window_full = any(
        all(term in _normalize(window) for term in terms) for window in windows
    )
    phrase = " ".join(query_terms)
    exact_phrase_match = any(phrase in window for window in windows)

    title = _normalize(hit.get("title", ""))
    title_all_terms = all(term in title for term in terms)
    title_any_term = any(term in title for term in terms)
    is_index = _is_index_document(hit.get("title", ""))

    score = coverage
    if same_window_full:
        score += 0.8
    if exact_phrase_match:
        score += 0.5
    if title_all_terms:
        score += 0.4
    elif title_any_term:
        score += 0.15
    score += min(len(windows), 3) * 0.05
    if coverage < 1.0:
        score -= 0.4
    if is_index:
        score -= 0.5
    if coverage <= 1.0 / len(terms) and len(terms) > 1:
        # Penalize hits that only mention one term among several.
        score -= 0.3
    return max(0.0, score)


def build_chunk_from_grep_hit(
    hit: dict, query_terms: list[str], rank: int = 0
) -> ChunkResult:
    """Build a ChunkResult from a grep hit so it enters the unified evidence chain."""
    doc_id = hit.get("yuque_id") or hit.get("id", "")
    body = _combine_grep_snippets(hit)
    score = score_grep_hit(hit, query_terms)
    matches = hit.get("matches", [{}])
    line_start = matches[0].get("line_start", 0)
    line_end = matches[-1].get("line_end", 0)
    reliable, _ = evaluate_grep_reliability(hit, query_terms)
    return ChunkResult(
        chunk_id=f"grep:{doc_id}:{line_start}:{line_end}",
        document_id=doc_id,
        title=hit.get("title", ""),
        content_snippet=body,
        source_url=hit.get("url", ""),
        keyword_score=score,
        final_score=score,
        retrieval_methods=("grep",),
        reliable=reliable,
        file_path=hit.get("path", ""),
        slug=hit.get("slug", ""),
        namespace=hit.get("namespace", ""),
    )


def grep_hits_to_search_results(
    hits: list[dict], query_terms: list[str]
) -> list[SearchResult]:
    """Convert ranked grep hits into SearchResult evidence objects."""
    ranked = sorted(
        hits, key=lambda hit: score_grep_hit(hit, query_terms), reverse=True
    )
    results: list[SearchResult] = []
    for i, hit in enumerate(ranked):
        chunk = build_chunk_from_grep_hit(hit, query_terms, rank=i)
        document = build_document_from_grep_hit(hit)
        results.append(
            SearchResult(
                source_id=f"G{i + 1}",
                document=document,
                score=chunk.final_score,
                chunk=chunk,
                keyword_score=chunk.keyword_score,
                retrieval_methods=("grep",),
                reliable=chunk.reliable,
            )
        )
    return results


def select_grounding_sources(
    sources: list[SearchResult], *, max_sources: int = 7
) -> list[SearchResult]:
    """Choose the best sources for the second-stage grounded prompt.

    Reliable sources come first, then the highest-scoring unreliable sources.
    """
    reliable = sorted(
        (s for s in sources if s.reliable), key=lambda s: s.score, reverse=True
    )
    unreliable = sorted(
        (s for s in sources if not s.reliable), key=lambda s: s.score, reverse=True
    )
    selected = reliable[:max_sources]
    if len(selected) < max_sources:
        selected.extend(unreliable[: max_sources - len(selected)])
    return selected


def merge_search_results(a: SearchResult, b: SearchResult) -> SearchResult:
    """Merge two results for the same document, keeping the best of each field."""
    methods = tuple(
        dict.fromkeys([*a.retrieval_methods, *b.retrieval_methods])
    )
    score = max(a.score, b.score)
    keyword_score = max(a.keyword_score, b.keyword_score)
    vector_score = max(a.vector_score, b.vector_score)
    reliable = a.reliable or b.reliable

    # Prefer the document with more body text (usually the read/full version).
    document = a.document
    if len(b.document.body) > len(a.document.body):
        document = b.document

    # Choose the richer chunk, then overlay merged metadata.
    if a.chunk and b.chunk:
        chunk = (
            a.chunk
            if len(a.chunk.content_snippet) >= len(b.chunk.content_snippet)
            else b.chunk
        )
        chunk = replace(
            chunk,
            retrieval_methods=methods,
            final_score=max(a.chunk.final_score, b.chunk.final_score),
            keyword_score=max(a.chunk.keyword_score, b.chunk.keyword_score),
            reliable=reliable,
        )
    else:
        chunk = a.chunk or b.chunk
        if chunk:
            chunk = replace(chunk, retrieval_methods=methods, reliable=reliable)

    return SearchResult(
        source_id=a.source_id,
        document=document,
        score=score,
        chunk=chunk,
        vector_score=vector_score,
        keyword_score=keyword_score,
        retrieval_methods=methods,
        reliable=reliable,
    )
