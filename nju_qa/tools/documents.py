"""Document navigation tools sharing the SQLite/Markdown data contract."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from ..doc_utils import (
    doc_record_to_public_dict,
    parse_yuque_doc_url,
    read_document_content,
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
        "按多个核心关键词搜索本地 Markdown 正文，返回上下文和可读取的 file_path。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "空格分隔的关键词"},
                "repo_filter": {"type": "string"},
            },
            "required": ["keywords"],
        }
    )

    async def _run(self, keywords: str, repo_filter: str = "", **_) -> dict:
        terms = [x for x in keywords.split() if x]
        hits = []
        for row in self.index.all_documents():
            if (
                repo_filter
                and repo_filter.casefold() not in row["repository"].casefold()
            ):
                continue
            text = row["title"] + "\n" + row["body"]
            found = [x for x in terms if x.casefold() in text.casefold()]
            if found:
                pos = min(text.casefold().find(x.casefold()) for x in found)
                public = doc_record_to_public_dict(row)
                hits.append(
                    {
                        **public,
                        "matched_keywords": found,
                        "snippet": text[max(0, pos - 120) : pos + 500],
                    }
                )
        return {"count": len(hits), "results": hits[:20]}


@dataclass
class ReadDocTool(_Tool):
    name: str = "read_doc"
    description: str = "通过 search 工具返回的 file_path 安全分页读取正文。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 12000},
            },
            "required": ["file_path"],
        }
    )

    async def _run(
        self, file_path: str, offset: int = 0, limit: int = 12000, **_
    ) -> dict:
        result = read_document_content(self.docs_root, file_path, offset, limit)
        if self.tracker:
            self.tracker.read_sources.add(file_path)
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
        result = {
            "count": 1,
            "document": results[0],
            **read_document_content(self.docs_root, results[0]["path"], offset, limit),
        }
        if self.tracker:
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
    description: str = "返回本地文档和向量索引数量。"
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    async def _run(self, **_) -> dict:
        return {
            "sqlite_documents": self.index.document_count(),
            "vector_documents": self.index.vector_count(),
        }
