"""Document navigation tools sharing the SQLite/Markdown data contract."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent
from ..doc_utils import (
    _load_cleaned_document_lines,
    doc_record_to_public_dict,
    parse_yuque_doc_url,
    read_document_content,
    read_document_lines,
)
from ..evidence import (
    document_from_index_row,
    evaluate_grep_reliability,
    score_grep_hit,
)


@dataclass
class _Tool(FunctionTool):
    index: Any = None
    docs_root: Any = None
    tracker: Any = None

    async def call(self, context, **kwargs):
        return await self._run(**kwargs)

    async def run(self, event: AstrMessageEvent, **kwargs):
        return await self._run(**kwargs)


@dataclass
class GrepLocalDocsTool(_Tool):
    name: str = "grep_local_docs"
    description: str = (
        "按空格分隔的关键词在本地的 Markdown 文档中逐行搜索，返回带行号的匹配片段。"
        "对于具体事实、名称、流程、时间等精确查询，优先使用本工具而不是向量检索。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "空格分隔的 1-4 个核心关键词，尽量用文档中可能出现的词",
                },
                "repo_filter": {"type": "string"},
                "context_lines": {
                    "type": "integer",
                    "default": 2,
                    "description": "匹配行前后保留的上下文行数",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["keywords"],
        }
    )

    async def _run(
        self,
        keywords: str,
        repo_filter: str = "",
        context_lines: int = 2,
        limit: int = 10,
        **_,
    ) -> dict:
        terms = [x for x in keywords.split() if x]
        hits = self._search(terms, repo_filter, context_lines, limit)
        # Fallback: long Chinese queries often fail exact substring matching.
        # Split into overlapping 2-character terms (e.g. "确认录取" → "确认 录取").
        if not hits:
            cleaned = keywords.strip().replace(" ", "")
            if len(cleaned) >= 4:
                split_terms = [
                    cleaned[i : i + 2] for i in range(0, len(cleaned) - 1, 2)
                ]
                hits = self._search(split_terms, repo_filter, context_lines, limit)
        if self.tracker:
            # Re-rank and register as unified evidence.
            hits = sorted(hits, key=lambda h: score_grep_hit(h, terms), reverse=True)
            reliable_count = sum(
                1 for h in hits if evaluate_grep_reliability(h, terms)[0]
            )
            logger.info(
                "NJU grep evidence: query=%r hits=%d reliable=%d tracked=%d",
                keywords,
                len(hits),
                reliable_count,
                len(self.tracker.sources),
            )
            self.tracker.add_grep_hits(hits, terms)
        return {"count": len(hits), "results": hits}

    def _search(
        self,
        terms: list[str],
        repo_filter: str,
        context_lines: int,
        limit: int,
    ) -> list[dict]:
        if not terms:
            return []
        hits: list[dict] = []
        cf = context_lines
        for row in self.index.all_documents():
            if (
                repo_filter
                and repo_filter.casefold() not in row["repository"].casefold()
            ):
                continue
            rel_path = row["path"]
            if not rel_path:
                continue
            try:
                lines = _load_cleaned_document_lines(self.docs_root, rel_path)
            except (OSError, ValueError):
                continue
            if not lines:
                continue

            matched_indices: set[int] = set()
            matched_terms: set[str] = set()
            lower_terms = [t.casefold() for t in terms]
            for i, line in enumerate(lines):
                line_lower = line.casefold()
                for term, term_lower in zip(terms, lower_terms):
                    if term_lower in line_lower:
                        matched_indices.add(i)
                        matched_terms.add(term)
            if not matched_indices:
                continue

            # Build contiguous windows around matched lines.
            sorted_idx = sorted(matched_indices)
            windows: list[tuple[int, int]] = []
            cur_start = cur_end = sorted_idx[0]
            for idx in sorted_idx[1:]:
                if idx - cur_end <= cf + 1:
                    cur_end = idx
                else:
                    windows.append((cur_start, cur_end))
                    cur_start = cur_end = idx
            windows.append((cur_start, cur_end))

            # Expand and merge windows.
            expanded = [
                (max(0, s - cf), min(len(lines) - 1, e + cf)) for s, e in windows
            ]
            merged: list[tuple[int, int]] = []
            for s, e in expanded:
                if merged and s <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))

            matches = []
            for s, e in merged:
                snippet = "\n".join(
                    f"{i + 1}: {lines[i]}" for i in range(s, e + 1)
                )
                matches.append(
                    {
                        "line_start": s + 1,
                        "line_end": e + 1,
                        "snippet": snippet,
                    }
                )

            public = doc_record_to_public_dict(row)
            hits.append(
                {
                    **public,
                    "score": round(
                        score_grep_hit(
                            {
                                **public,
                                "matched_keywords": sorted(matched_terms),
                                "matches": matches,
                            },
                            terms,
                        ),
                        3,
                    ),
                    "matched_keywords": sorted(matched_terms),
                    "matches": matches[:3],
                }
            )
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:limit]


@dataclass
class ReadDocTool(_Tool):
    name: str = "read_doc"
    description: str = (
        "通过 search/grep 工具返回的 file_path 安全读取正文。"
        "支持按字符分页（offset/limit）或按行号范围（start_line/end_line）读取。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 12000},
                "start_line": {
                    "type": "integer",
                    "description": "与 end_line 配合时按行号读取，优先级高于 offset",
                },
                "end_line": {"type": "integer"},
                "context_lines": {"type": "integer", "default": 0},
            },
            "required": ["file_path"],
        }
    )

    async def _run(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 12000,
        start_line: int | None = None,
        end_line: int | None = None,
        context_lines: int = 0,
        **_,
    ) -> dict:
        try:
            if start_line is not None or end_line is not None:
                result = read_document_lines(
                    self.docs_root,
                    file_path,
                    start_line=start_line or 0,
                    end_line=end_line,
                    context_lines=context_lines,
                )
            else:
                result = read_document_content(
                    self.docs_root, file_path, offset, limit
                )
        except ValueError as exc:
            return {"error": str(exc), "file_path": file_path}
        if self.tracker:
            resolved_path = result.get("file_path", file_path)
            row = next(
                (
                    r
                    for r in self.index.all_documents()
                    if r["path"] == resolved_path
                ),
                None,
            )
            if row is not None:
                document = document_from_index_row(row, result["content"])
                self.tracker.add_read_document(document, result["content"])
            else:
                self.tracker.read_sources.add(resolved_path)
                self.tracker.record_read_content(result["content"])
        return result


@dataclass
class SearchDocsTool(_Tool):
    name: str = "search_docs"
    description: str = (
        "按标题、slug、语雀 ID 或 URL 查询 SQLite 元数据；同名标题会保留全部候选。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "yuque_id": {"type": "string"},
                "slug": {"type": "string"},
                "url": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        }
    )

    async def _run(
        self,
        query: str = "",
        yuque_id: str = "",
        slug: str = "",
        url: str = "",
        limit: int = 20,
        offset: int = 0,
        **_,
    ) -> dict:
        rows = self.index.find(
            query, yuque_id=yuque_id, slug=slug, url=url, limit=limit, offset=offset
        )
        return {
            "count": len(rows),
            "results": [doc_record_to_public_dict(row) for row in rows],
        }


@dataclass
class GetDocDetailsTool(_Tool):
    name: str = "get_doc_details"
    description: str = "通过 title、file_path、yuque_id 或 URL 获取元数据和可选正文；同名标题不擅自选择。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "file_path": {"type": "string"},
                "yuque_id": {"type": "string"},
                "url": {"type": "string"},
                "include_content": {"type": "boolean", "default": False},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 12000},
            },
            "required": [],
        }
    )

    async def _run(
        self,
        title: str = "",
        file_path: str = "",
        yuque_id: str = "",
        url: str = "",
        include_content: bool = False,
        offset: int = 0,
        limit: int = 12000,
        **_,
    ) -> dict:
        rows = [
            row
            for row in self.index.find(title, yuque_id=yuque_id, url=url, limit=100)
            if not file_path or row["path"] == file_path
        ]
        results = [doc_record_to_public_dict(row) for row in rows]
        if len(results) != 1 or not include_content:
            return {"count": len(results), "results": results}
        try:
            result = {
                "count": 1,
                "document": results[0],
                **read_document_content(self.docs_root, results[0]["path"], offset, limit),
            }
        except ValueError as exc:
            return {
                "count": 1,
                "document": results[0],
                "error": str(exc),
                "file_path": results[0]["path"],
            }
        if self.tracker:
            row = next(
                (r for r in rows if r["path"] == results[0]["path"]),
                None,
            )
            if row is not None:
                document = document_from_index_row(row, result["content"])
                self.tracker.add_read_document(document, result["content"])
            else:
                self.tracker.read_sources.add(results[0]["path"])
                self.tracker.record_read_content(result["content"])
        return result


@dataclass
class ParseYuqueUrlTool(_Tool):
    name: str = "parse_yuque_url"
    description: str = (
        "解析带 query 或 anchor 的语雀文档链接，并定位配置范围内的本地文档。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        }
    )

    async def _run(self, url: str, **_) -> dict:
        parsed = parse_yuque_doc_url(url)
        if not parsed:
            return {"count": 0, "results": []}
        namespace, slug = parsed
        rows = [
            r
            for r in self.index.find(slug=slug, limit=20)
            if r["namespace"] == namespace
        ]
        return {
            "count": len(rows),
            "results": [doc_record_to_public_dict(row) for row in rows],
        }


@dataclass
class ListKnowledgeBasesTool(_Tool):
    name: str = "list_knowledge_bases"
    description: str = "列出已同步知识库及文档数量。"
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    async def _run(self, **_) -> dict:
        groups = {}
        for row in self.index.all_documents():
            groups[row["namespace"]] = groups.get(row["namespace"], 0) + 1
        return {
            "knowledge_bases": [
                {"namespace": k, "documents": v} for k, v in groups.items()
            ]
        }


@dataclass
class ListRepoDocsTool(_Tool):
    name: str = "list_repo_docs"
    description: str = "列出指定知识库的标题、slug 和 file_path。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"namespace": {"type": "string"}},
            "required": ["namespace"],
        }
    )

    async def _run(self, namespace: str, **_) -> dict:
        return {
            "documents": [
                doc_record_to_public_dict(r)
                for r in self.index.all_documents()
                if r["namespace"] == namespace
            ]
        }


@dataclass
class DocStatsTool(_Tool):
    name: str = "doc_stats"
    description: str = "返回本地文档、chunk 和向量索引数量。"
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    async def _run(self, **_) -> dict:
        result = {
            "sqlite_documents": self.index.document_count(),
            "vector_documents": self.index.vector_count(),
        }
        if hasattr(self, "chunk_store") and self.chunk_store is not None:
            result["chunk_documents"] = self.chunk_store.documents_with_chunks()
            result["chunk_total"] = self.chunk_store.chunk_count()
        return result
