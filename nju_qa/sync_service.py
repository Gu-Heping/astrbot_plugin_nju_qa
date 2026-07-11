"""Synchronizes Yuque documents into local Markdown files, SQLite metadata and chunk indexes."""

from __future__ import annotations
import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from .chunk_indexer import ChunkIndexer
from .config import PluginConfig
from .document_index import DocumentIndex
from .document_store import DocumentStore
from .models import Document, SyncResult
from .retriever import HybridRetriever
from .yuque_client import YuqueClient


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class SyncService:
    def __init__(
        self,
        config: PluginConfig,
        client: YuqueClient,
        store: DocumentStore,
        index: DocumentIndex,
        chunk_store=None,
        vector_index=None,
        embed=None,
    ):
        self.config = config
        self.client = client
        self.store = store
        self.index = index
        self.chunk_store = chunk_store
        self.vector_index = vector_index
        self._custom_embed = embed
        self._sync_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self.running = False
        self._last_index_error: str | None = None

    def status_text(self) -> str:
        state = "进行中" if self.running else "空闲"
        last_sync = self.index.get_state("last_sync") or "从未"
        last_index = self.index.get_state("last_index") or "从未"
        last_index_result = self.index.get_state("last_index_result") or "无"
        md_count = len(list(self.store.root.rglob("*.md")))
        sqlite_count = self.index.document_count()
        chunk_docs = (
            self.chunk_store.documents_with_chunks() if self.chunk_store else 0
        )
        chunk_total = self.chunk_store.chunk_count() if self.chunk_store else 0
        vector_count = self.vector_index.count() if self.vector_index else 0
        embedding_ready = bool(
            self.config.embedding_api_key and self.config.embedding_base_url
        )
        return (
            f"同步状态：{state}\n"
            f"上次同步：{last_sync}\n"
            f"最近索引：{last_index}\n"
            f"最近索引结果：{last_index_result}\n"
            f"Markdown 文档数：{md_count}\n"
            f"SQLite 文档数：{sqlite_count}\n"
            f"已建立 chunk 的文档数：{chunk_docs}\n"
            f"chunk 总数：{chunk_total}\n"
            f"已向量化 chunk 数：{vector_count}\n"
            f"Embedding 可用：{embedding_ready}\n"
            f"Embedding 模型：{self.config.embedding_model}\n"
            f"索引失败：{self._last_index_error or '无'}"
        )

    async def sync_all(self) -> SyncResult:
        if not self.config.yuque_token:
            raise ValueError("未配置 yuque_token")
        if not self.config.repositories:
            raise ValueError("未配置 yuque_repositories")
        async with self._sync_lock:
            self.running = True
            result = SyncResult()
            try:
                for repo in self.config.repositories:
                    await self._sync_repository(repo.namespace, repo.name, result)
                async with self._index_lock:
                    index_result = await self._index_changed_documents(result)
                self.index.set_state(
                    "last_sync", datetime.now(timezone.utc).isoformat()
                )
                self.index.set_state(
                    "last_index",
                    datetime.now(timezone.utc).isoformat(),
                )
                self.index.set_state("last_index_result", str(index_result))
                result.chunks_indexed = index_result.get("chunks", 0)
                result.chunks_failed = index_result.get("failed_documents", 0)
                return result
            finally:
                self.running = False

    async def _sync_repository(
        self, namespace: str, name: str, result: SyncResult
    ) -> None:
        repo = await self.client.get_repo(namespace)
        toc = await self.client.get_toc(namespace)
        by_uuid = {x.get("uuid"): x for x in toc if x.get("uuid")}
        paths = self._toc_paths(toc, by_uuid)
        seen = set()
        used = set()
        for node in toc:
            if node.get("type") not in {"DOC", "SHEET"} or not node.get("id"):
                continue
            doc_id = str(node["id"])
            seen.add(doc_id)
            try:
                slug = str(node.get("url") or node.get("slug") or "")
                if not slug:
                    result.skipped += 1
                    continue
                detail = await self.client.get_document(namespace, slug)
                title = str(detail.get("title") or node.get("title") or slug)
                parents = paths.get(node.get("uuid"), [])
                existing = next(
                    (r for r in self.index.all_documents() if r["yuque_id"] == doc_id),
                    None,
                )
                path = self.store.path_for(namespace, parents, title, doc_id, used)
                used.add(path)
                url = str(
                    detail.get("url") or f"https://www.yuque.com/{namespace}/{slug}"
                )
                doc = Document(
                    doc_id,
                    title,
                    str(repo.get("name") or name or namespace),
                    namespace,
                    str(detail.get("slug") or slug),
                    url,
                    str(detail.get("created_at") or ""),
                    str(detail.get("updated_at") or ""),
                    str(detail.get("body") or detail.get("content") or ""),
                    path,
                )
                if existing and Path(existing["path"]) != path:
                    self.store.remove(Path(existing["path"]))
                self.store.write(doc)
                self.index.upsert(doc)
                # Metadata-only updates for renamed or moved documents.
                if existing and self.chunk_store is not None:
                    if existing["title"] != title or existing["path"] != str(path):
                        self.chunk_store.update_document_metadata(
                            doc_id,
                            title=title,
                            repository=doc.repository,
                            namespace=namespace,
                            slug=doc.slug,
                            file_path=str(path),
                            source_url=url,
                            updated_at=doc.updated_at,
                        )
                result.succeeded += 1
            except Exception:
                result.failed += 1
        for row in self.index.delete_missing(namespace, seen):
            self.store.remove(Path(row["path"]))
            if self.chunk_store is not None:
                self.chunk_store.delete_document(row["yuque_id"])
            if self.vector_index is not None:
                self.vector_index.delete_document(row["yuque_id"])
            result.deleted += 1

    async def _index_changed_documents(self, result: SyncResult) -> dict:
        """Index chunks for new or updated documents; skip unchanged bodies."""
        if self.chunk_store is None or self.vector_index is None:
            return {"chunks": 0, "failed_documents": 0}
        indexer = ChunkIndexer(
            self.chunk_store,
            self.vector_index,
            self._embed,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        total = failed = 0
        errors: list[str] = []
        for row in self.index.all_documents():
            doc_id = str(row["yuque_id"])
            new_hash = _body_hash(row["body"])
            existing = self.chunk_store.get_document_chunks(doc_id)
            if existing and all(c.content_hash == new_hash for c in existing):
                # Body unchanged; still make sure metadata is current.
                self.chunk_store.update_document_metadata(
                    doc_id,
                    title=row["title"],
                    repository=row["repository"],
                    namespace=row["namespace"],
                    slug=row["slug"],
                    file_path=row["path"] or "",
                    source_url=row["url"],
                    updated_at=row["updated_at"],
                )
                continue
            res = await indexer.index_document(row)
            total += res["chunks"]
            if res["error"]:
                failed += 1
                errors.append(f"{doc_id}: {res['error']}")
        if errors:
            self._last_index_error = "; ".join(errors[:5])
        return {"chunks": total, "failed_documents": failed, "errors": errors}

    async def _embed(self, text: str) -> list[float] | None:
        if self._custom_embed is not None:
            try:
                return await self._custom_embed(text)
            except Exception as exc:
                self._last_index_error = f"embedding failed: {type(exc).__name__}"
                return None
        if not self.config.embedding_api_key or not self.config.embedding_base_url:
            return None
        return await HybridRetriever(self.index, self.config).embed_text(text)

    def _toc_paths(self, toc, by_uuid):
        output = {}

        def walk(node):
            parents = []
            current = node
            while current and current.get("parent_uuid"):
                current = by_uuid.get(current.get("parent_uuid"))
                if current and current.get("type") not in {"DOC", "SHEET"}:
                    parents.append(str(current.get("title") or "folder"))
            return list(reversed(parents))

        for node in toc:
            output[node.get("uuid")] = walk(node)
        return output

    async def rebuild_index(self) -> str:
        async with self._sync_lock:
            async with self._index_lock:
                from .chunk_indexer import ChunkIndexer

                indexer = ChunkIndexer(
                    self.chunk_store,
                    self.vector_index,
                    self._embed,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                )
                result = await indexer.rebuild(self.index.all_documents())
                self._last_index_error = (
                    "; ".join(result.get("errors", [])) or None
                )
                self.index.set_state(
                    "last_index",
                    datetime.now(timezone.utc).isoformat(),
                )
                self.index.set_state(
                    "last_index_result",
                    f"chunks={result['chunks']} failed_docs={result['failed_documents']}",
                )
        return (
            f"向量索引重建完成：{result['chunks']} 个 chunk，"
            f"失败文档 {result['failed_documents']}。"
        )
