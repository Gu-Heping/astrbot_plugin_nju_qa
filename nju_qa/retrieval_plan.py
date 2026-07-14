"""Retrieval planning and evidence coverage classification.

The planner splits a user question into sub-questions/entities, scopes each one
to the knowledge-base structure, and decides which tool is most appropriate.
After retrieval, coverage classification ensures that entity-specific questions
are only answered with DIRECT evidence (the entity is mentioned in a matched
source), not with generic campus background material.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

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
    "什么 怎么 哪里 哪儿 多少 何时 几时 何地 为什么 吗 呢 吧".split()
)


def _split_subquestions(query: str) -> list[str]:
    """Split a query into sub-questions at punctuation and conjunctions."""
    parts = re.split(r"[，。；；？?；；]|\s+(?:和|与|及|以及|还是|或者)\s+", query)
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


def _core_terms(query: str) -> list[str]:
    """Return the query with generic and interrogative words stripped out."""
    removable = sorted(
        _GENERIC_CAMPUS_WORDS | _INTERROGATIVE_WORDS,
        key=len,
        reverse=True,
    )
    cleaned = query
    for word in removable:
        cleaned = cleaned.replace(word, " ")
    terms = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]
    return terms


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
    A list of :class:`RetrievalNeed`.  If the query cannot be split, a single
    need covering the whole query is returned.
    """
    candidates = _collect_entity_candidates(rows)
    subquestions = _split_subquestions(query)
    plan: list[RetrievalNeed] = []

    for sub in subquestions:
        # Entity terms are document titles / path segments that appear verbatim
        # in the sub-question and are not generic campus words.
        entity_terms = [
            cand
            for cand in candidates
            if cand in sub
            and cand not in _GENERIC_CAMPUS_WORDS
            and len(cand) >= 2
        ]
        # Question terms are non-generic, non-interrogative words from the
        # sub-question.  Core terms come from stripping generic words; extracted
        # terms catch words like "大一" that are not in the candidate set.
        core_terms = _core_terms(sub)
        extracted_terms = [
            term
            for term in _extract_terms(sub)
            if term not in _GENERIC_CAMPUS_WORDS
            and term not in _INTERROGATIVE_WORDS
            and len(term) >= 2
            and term not in core_terms
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
        question_terms = list(dict.fromkeys(core_terms + candidate_terms + extracted_terms))
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


def _source_text(source: Any) -> str:
    """Return a searchable text representation of a source."""
    parts: list[str] = []
    document = getattr(source, "document", None)
    chunk = getattr(source, "chunk", None)
    if document is not None:
        parts.append(document.title)
        if document.path:
            parts.append(str(document.path))
        parts.append(document.body)
    if chunk is not None:
        parts.append(chunk.title)
        parts.append(chunk.file_path or "")
        parts.append(chunk.content_snippet)
    return "\n".join(parts)


def _score_source_for_need(need: RetrievalNeed, source: Any) -> tuple[bool, bool]:
    """Return (direct, background) flags for a source against a need."""
    text = _source_text(source).casefold()
    entity_terms = [t.casefold() for t in need.entity_terms]
    core_terms = [t.casefold() for t in need.core_terms]
    question_terms = [t.casefold() for t in need.question_terms]

    if entity_terms:
        if all(term in text for term in entity_terms):
            return True, True
        if question_terms and any(term in text for term in question_terms):
            return False, True
        return False, False

    if core_terms and all(term in text for term in core_terms):
        return True, True
    if question_terms and any(term in text for term in question_terms):
        return False, True

    return False, False


def classify_coverage(
    need: RetrievalNeed, sources: list[Any]
) -> CoverageResult:
    """Classify whether ``sources`` provide DIRECT, BACKGROUND, or no evidence."""
    direct: list[Any] = []
    background: list[Any] = []
    for source in sources:
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
    plan: list[RetrievalNeed], sources: list[Any]
) -> list[CoverageResult]:
    """Run coverage classification for every need in a plan."""
    return [classify_coverage(need, sources) for need in plan]


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
        "覆盖要求：每个子问题必须获得 DIRECT 证据（证据中明确出现对应实体/关键词）"
        "才能用于回答；只有 BACKGROUND 证据时，不能回答实体特定部分。"
    )
    return "\n".join(lines)
