"""Retrieval planning and evidence coverage classification.

The planner splits a user question into sub-questions, extracts campus entities
from the query itself (unknown entities are kept so the retrieval layer can
search for them), and scopes each sub-question to the knowledge-base structure.

After retrieval, coverage classification ensures that entity-specific questions
are only answered with DIRECT evidence: a *reliable* source whose supporting
window contains both the entity and the core question condition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .entities import (
    EntityMention,
    extract_entities,
    resolve_entities,
)
from .evidence import _ACTION_SYNONYMS, _NO_ANSWER_MARKERS
from .keyword_index import _extract_query_terms
from .knowledge_structure import _normalize_path, _path_segments


class CoverageStatus(str, Enum):
    """Evidence relationship for a single sub-question."""

    DIRECT = "DIRECT"
    BACKGROUND = "BACKGROUND"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class RetrievalNeed:
    """One planned retrieval step for a sub-question or entity."""

    question: str
    entity_terms: list[str] = field(default_factory=list)
    question_terms: list[str] = field(default_factory=list)
    core_terms: list[str] = field(default_factory=list)
    scope_namespace: str = ""
    scope_path_prefix: str = ""
    preferred_tool: str = "search_knowledge_base"
    evidence_required: bool = True


@dataclass
class CoverageResult:
    """Coverage classification for one RetrievalNeed."""

    need: RetrievalNeed
    status: CoverageStatus
    sources: list[Any] = field(default_factory=list)


# Terms that make a sub-question point to precise facts and therefore favor
# keyword/grep retrieval over pure vector search.
_GROUNDED_TRIGGERS = frozenset(
    "哪里 地点 地址 时间 流程 要求 条件 多少 费用 怎么办 怎么办理 需要带什么 "
    "联系方式 电话 网址 网站 入口 截止日期 期限 何时 何地".split()
)

# Generic campus words; presence alone does not make a sub-question entity-specific.
_GENERIC_CAMPUS_WORDS = frozenset(
    "南京大学 南大 校园 校区 学生 新生 老生 本科生 研究生 教务 课程 学分 考试 "
    "成绩 培养方案 选课 退课 补考 重修 图书馆 宿舍 食堂 交通 奖助 奖学金 "
    "助学金 入学 报到 毕业 就业 户口 档案 党组织 团组织 军训 体检 缴费 "
    "学费 住宿 校园卡 学生证 身份证".split()
)

_INTERROGATIVE_WORDS = frozenset(
    "什么 怎么 哪里 在哪儿 在哪 哪儿 多少 何时 几时 何地 为什么 吗 呢 吧 "
    "如何 怎样 怎么办".split()
)

_ACTION_WORDS: frozenset[str] = frozenset(
    {a for terms in _ACTION_SYNONYMS.values() for a in terms}
    | set(_ACTION_SYNONYMS.keys())
)


def _find_spans(query: str, words: set[str]) -> list[tuple[int, int]]:
    """Return non-overlapping (start, end) spans of ``words`` in ``query``."""
    spans: list[tuple[int, int]] = []
    for word in sorted({w for w in words if w}, key=len, reverse=True):
        for match in re.finditer(re.escape(word), query):
            spans.append((match.start(), match.end()))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start < prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _protected_spans(query: str, entity_terms: list[str]) -> list[tuple[int, int]]:
    """Spans of generic, interrogative, action and entity words in ``query``."""
    words = set(_GENERIC_CAMPUS_WORDS | _INTERROGATIVE_WORDS | _ACTION_WORDS)
    words.update(entity_terms)
    return _find_spans(query, words)


def _is_cross_boundary(
    query: str, term: str, spans: list[tuple[int, int]]
) -> bool:
    """Return True when ``term`` straddles or is fully inside a protected span.

    Terms that exactly match a protected span are allowed; they are filtered
    elsewhere (generic/interrogative words are removed, action words are kept).
    """
    if not spans:
        return False
    for match in re.finditer(re.escape(term), query):
        start, end = match.start(), match.end()
        for span_start, span_end in spans:
            if start == span_start and end == span_end:
                continue
            if start < span_end and end > span_start:
                return True
    return False


def _normalize_marker(text: str) -> str:
    return text.casefold()
    return text.casefold()


_NO_ANSWER_RE = re.compile(
    "|".join(re.escape(_normalize_marker(m)) for m in _NO_ANSWER_MARKERS),
    re.IGNORECASE,
)


def _split_subquestions(query: str) -> list[str]:
    """Split a query into sub-questions at punctuation and conjunctions."""
    parts = re.split(r"[，。；；？?；；]\s*|\s+(?:和|与|及|以及|还是|或者)\s+", query)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned or [query.strip()]


def _collect_entity_candidates(rows: list[Any]) -> set[str]:
    """Build a set of likely entity names from document titles and path segments."""
    candidates: set[str] = set()
    for row in rows:
        title = (row.get("title") if hasattr(row, "get") else row["title"]) or ""
        if title:
            candidates.add(title)
        rel_path = _normalize_path(
            row.get("path") if hasattr(row, "get") else row["path"]
        )
        for segment in _path_segments(rel_path):
            if segment.lower() in {"归档", "00_index", "index"}:
                continue
            if len(segment) >= 2:
                candidates.add(segment)
                candidates.add(Path(segment).stem)
    return candidates


def _extract_terms(text: str) -> list[str]:
    """Extract search-friendly terms from a query, including Chinese bigrams."""
    return _extract_query_terms(text)


def _find_matching_path_prefix(
    entity_terms: list[str], rows: list[Any]
) -> tuple[str, str]:
    """Return (namespace, path_prefix) when an entity term matches a path segment.

    The path_prefix is returned relative to its namespace.
    """
    for term in entity_terms:
        for row in rows:
            rel_path = _normalize_path(
                row.get("path") if hasattr(row, "get") else row["path"]
            )
            segments = _path_segments(rel_path)
            if not segments:
                continue
            namespace = segments[0]
            for i, segment in enumerate(segments[1:], start=1):
                if term in segment:
                    prefix = "/".join(segments[1:i]) if i > 1 else ""
                    return namespace, prefix
    return "", ""


def _is_generic_or_interrogative(term: str) -> bool:
    """Return True when a term is part of a generic campus or interrogative word."""
    low = term.casefold()
    for generic in _GENERIC_CAMPUS_WORDS | _INTERROGATIVE_WORDS:
        if low in generic.casefold() or generic.casefold() in low:
            return True
    return False


def _core_terms(query: str, entity_terms: list[str] | None = None) -> list[str]:
    """Return core question terms with generic, interrogative and entity words removed."""
    spans = _protected_spans(query, entity_terms or [])
    terms = [
        term
        for term in _extract_terms(query)
        if len(term) >= 2
        and not _is_generic_or_interrogative(term)
        and not (entity_terms and any(term in e or e in term for e in entity_terms))
        and not _is_cross_boundary(query, term, spans)
    ]
    return list(dict.fromkeys(terms))


def _entity_texts(mentions: list[EntityMention]) -> list[str]:
    """Return unique entity mention texts in query order."""
    seen: set[str] = set()
    result: list[str] = []
    for m in mentions:
        if m.text not in seen:
            seen.add(m.text)
            result.append(m.text)
    return result


def build_retrieval_plan(query: str, rows: list[Any]) -> list[RetrievalNeed]:
    """Create a retrieval plan for ``query`` based on document metadata.

    Parameters
    ----------
    query:
        Original user question.
    rows:
        Document metadata rows (SQLite rows or dicts) with ``title`` and ``path``.

    Returns
    -------
    A list of :class:`RetrievalNeed`.  Entity mentions are extracted from the
    query itself; unknown entities are kept and searched for, not silently dropped.
    """
    candidates = _collect_entity_candidates(rows)
    mentions = resolve_entities(extract_entities(query), rows)
    subquestions = _split_subquestions(query)
    plan: list[RetrievalNeed] = []

    for sub in subquestions:
        sub_mentions = [m for m in mentions if m.text in sub]
        entity_terms = _entity_texts(sub_mentions)

        core_terms = _core_terms(sub, entity_terms)
        spans = _protected_spans(sub, entity_terms)
        extracted_terms = [
            term
            for term in _extract_terms(sub)
            if term not in _GENERIC_CAMPUS_WORDS
            and term not in _INTERROGATIVE_WORDS
            and len(term) >= 2
            and term not in core_terms
            and term not in entity_terms
            and not _is_cross_boundary(sub, term, spans)
        ]
        candidate_terms = [
            cand
            for cand in candidates
            if cand in sub
            and cand not in _GENERIC_CAMPUS_WORDS
            and cand not in _INTERROGATIVE_WORDS
            and len(cand) >= 2
            and cand not in entity_terms
            and cand not in core_terms
        ]
        question_terms = list(
            dict.fromkeys(core_terms + candidate_terms + extracted_terms)
        )

        preferred_tool = "search_knowledge_base"
        if any(trigger in sub for trigger in _GROUNDED_TRIGGERS):
            preferred_tool = "grep_local_docs"

        scope_namespace, scope_path_prefix = _find_matching_path_prefix(
            entity_terms, rows
        )

        plan.append(
            RetrievalNeed(
                question=sub,
                entity_terms=entity_terms,
                question_terms=question_terms,
                core_terms=core_terms,
                scope_namespace=scope_namespace,
                scope_path_prefix=scope_path_prefix,
                preferred_tool=preferred_tool,
                evidence_required=True,
            )
        )
    return plan


def _source_windows(source: Any) -> list[str]:
    """Return the supporting window(s) that can be used for coverage checks."""
    text = ""
    chunk = getattr(source, "chunk", None)
    document = getattr(source, "document", None)
    if chunk is not None and chunk.content_snippet:
        text = chunk.content_snippet
    elif document is not None and document.body:
        text = document.body
    if not text:
        return []
    # Grep snippets often concatenate multiple match windows with blank lines.
    windows = [w.strip() for w in text.split("\n\n") if w.strip()]
    return windows or [text.strip()]


def _window_has_no_answer(window: str) -> bool:
    """Return True when a window contains an explicit no-answer marker."""
    return bool(_NO_ANSWER_RE.search(window.casefold()))


def _window_covers_need(window: str, need: RetrievalNeed) -> bool:
    """Return True when a supporting window directly covers a need.

    For entity-specific needs, a DIRECT window must contain all entity terms and
    at least one core question term.  For non-entity needs, all core terms are
    required (or any question term when no core terms remain).
    """
    if _window_has_no_answer(window):
        return False
    norm = window.casefold()
    entity_terms = [t.casefold() for t in need.entity_terms]
    core_terms = [t.casefold() for t in need.core_terms]

    if entity_terms and not all(term in norm for term in entity_terms):
        return False
    if core_terms:
        if entity_terms:
            return any(term in norm for term in core_terms)
        return all(term in norm for term in core_terms)
    if not entity_terms and not core_terms:
        return any(t.casefold() in norm for t in need.question_terms)
    return True


def _window_is_background(window: str, need: RetrievalNeed) -> bool:
    """Return True when a window is relevant but not directly answering the need."""
    if _window_has_no_answer(window):
        return False
    if _window_covers_need(window, need):
        return False
    norm = window.casefold()
    terms = [t.casefold() for t in need.core_terms or need.question_terms]
    return any(term in norm for term in terms)


def _score_source_for_need(
    need: RetrievalNeed, source: Any
) -> tuple[bool, bool]:
    """Return (direct, background) flags for a reliable source against a need."""
    windows = _source_windows(source)
    if not windows:
        return False, False

    for window in windows:
        if _window_covers_need(window, need):
            return True, True

    for window in windows:
        if _window_is_background(window, need):
            return False, True

    return False, False


def classify_coverage(
    need: RetrievalNeed,
    sources: list[Any],
    *,
    require_reliable: bool = True,
) -> CoverageResult:
    """Classify whether ``sources`` provide DIRECT, BACKGROUND, or no evidence.

    By default only reliable sources are considered, because unreliable hits
    (e.g. index documents or no-answer QA entries) must not back an answer.
    """
    direct: list[Any] = []
    background: list[Any] = []
    for source in sources:
        if require_reliable and not getattr(source, "reliable", False):
            continue
        is_direct, is_background = _score_source_for_need(need, source)
        if is_direct:
            direct.append(source)
        elif is_background:
            background.append(source)

    if direct:
        return CoverageResult(need, CoverageStatus.DIRECT, direct)
    if background:
        return CoverageResult(need, CoverageStatus.BACKGROUND, background)
    return CoverageResult(need, CoverageStatus.UNSUPPORTED, [])


def check_coverage(
    plan: list[RetrievalNeed],
    sources: list[Any],
    *,
    require_reliable: bool = True,
) -> list[CoverageResult]:
    """Run coverage classification for every need in a plan."""
    return [
        classify_coverage(need, sources, require_reliable=require_reliable)
        for need in plan
    ]


def format_plan(plan: list[RetrievalNeed]) -> str:
    """Render a retrieval plan as instructions for the LLM."""
    lines = ["检索计划："]
    for i, need in enumerate(plan, 1):
        scope = need.scope_namespace
        if need.scope_path_prefix:
            scope += f"/{need.scope_path_prefix}"
        entity_note = ""
        if need.entity_terms:
            entity_note = f" 实体：{', '.join(need.entity_terms)}"
        lines.append(
            f"{i}. {need.question}（推荐工具：{need.preferred_tool}"
            f"，作用域：{scope or '全部'}{entity_note}）"
        )
    lines.append(
        "覆盖要求：每个子问题必须获得 DIRECT 证据（可靠来源的同一支持窗口中"
        "同时出现对应实体和核心问题条件）才能用于回答；只有 BACKGROUND 证据时，"
        "不能回答实体特定部分。"
    )
    return "\n".join(lines)


def format_coverage_report(coverage: list[CoverageResult]) -> str:
    """Human-readable summary of coverage for prompts or diagnostics."""
    lines = ["证据覆盖情况："]
    for cov in coverage:
        scope = ""
        if cov.need.entity_terms:
            scope = f" 实体：{', '.join(cov.need.entity_terms)}"
        lines.append(
            f"- {cov.need.question} → {cov.status.value}"
            f"（支持来源 {len(cov.sources)} 个）{scope}"
        )
    return "\n".join(lines)
