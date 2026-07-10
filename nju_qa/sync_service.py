from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from .config import PluginConfig
from .document_index import DocumentIndex
from .document_store import DocumentStore
from .models import Document, SyncResult
from .retriever import HybridRetriever
from .yuque_client import YuqueClient


class SyncService:
    def __init__(
        self,
        config: PluginConfig,
        client: YuqueClient,
        store: DocumentStore,
        index: DocumentIndex,
    ):
        self.config, self.client, self.store, self.index = config, client, store, index
        self._sync_lock = asyncio.Lock()
        self._index_lock = asyncio.Lock()
        self.running = False

    def status_text(self) -> str:
        state = "进行中" if self.running else "空闲"
        last = self.index.get_state("last_sync") or "从未"
        return f"同步状态：{state}\n上次同步：{last}\n文档数：{len(self.index.all_documents())}"

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
                    await self._rebuild_embeddings()
                self.index.set_state(
                    "last_sync", datetime.now(timezone.utc).isoformat()
                )
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
                # Recompute this on every run so TOC moves and title renames are
                # reflected locally while the stable Yuque ID preserves identity.
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
                result.succeeded += 1
            except Exception:
                result.failed += 1
        for row in self.index.delete_missing(namespace, seen):
            self.store.remove(Path(row["path"]))
            result.deleted += 1

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
        # Acquire both locks in the same order as sync: a rebuild must never
        # read/write embeddings while synchronization is changing documents.
        async with self._sync_lock:
            async with self._index_lock:
                count = await self._rebuild_embeddings()
        return f"向量索引重建完成：{count} 篇文档。"

    async def _rebuild_embeddings(self) -> int:
        return await HybridRetriever(self.index, self.config).rebuild_embeddings()
