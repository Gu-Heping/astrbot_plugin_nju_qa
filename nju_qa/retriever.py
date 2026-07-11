from __future__ import annotations
import httpx
from .config import PluginConfig
from .document_index import DocumentIndex
from .models import Document, SearchResult


class HybridRetriever:
    def __init__(self, index: DocumentIndex, config: PluginConfig):
        self.index, self.config = index, config

    async def _embed(self, text: str) -> list[float] | None:
        if not self.config.embedding_api_key or not self.config.embedding_base_url:
            return None
        url = self.config.embedding_base_url.rstrip("/") + "/embeddings"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {self.config.embedding_api_key}"},
                json={"model": self.config.embedding_model, "input": text[:8000]},
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    @staticmethod
    def _doc(row) -> Document:
        from pathlib import Path

        return Document(
            *[
                row[k]
                for k in (
                    "yuque_id",
                    "title",
                    "repository",
                    "namespace",
                    "slug",
                    "url",
                    "created_at",
                    "updated_at",
                    "body",
                )
            ],
            Path(row["path"]) if row["path"] else None,
        )

    async def search(self, query: str) -> list[SearchResult]:
        report = await self.debug_search(query)
        return report["selected"]

    async def debug_search(self, query: str) -> dict:
        merged: dict[str, float] = {}
        keyword = self.index.keyword(query, self.config.retrieval_top_k * 4)
        for row, score in keyword:
            merged[row["yuque_id"]] = max(
                merged.get(row["yuque_id"], 0), min(1, score / 3)
            )
        vector = await self._embed(query)
        vector_rows = []
        if vector:
            vector_rows = self.index.vector(vector, self.config.retrieval_top_k * 2)
            for row, score in vector_rows:
                merged[row["yuque_id"]] = max(merged.get(row["yuque_id"], 0), score)
        lookup = {r["yuque_id"]: r for r in self.index.all_documents()}
        selected = []
        for ident, score in sorted(merged.items(), key=lambda i: i[1], reverse=True):
            if score >= self.config.score_threshold:
                selected.append(
                    SearchResult(
                        f"S{len(selected) + 1}", self._doc(lookup[ident]), score
                    )
                )
            if len(selected) >= self.config.retrieval_top_k:
                break
        return {
            "mode": "hybrid" if vector else "keyword",
            "embedding_available": bool(vector),
            "keyword": [(r["title"], s) for r, s in keyword],
            "vector": [(r["title"], s) for r, s in vector_rows],
            "selected": selected,
            "threshold": self.config.score_threshold,
            "vector_count": self.index.vector_count(),
        }

    def debug_text(self, report: dict) -> str:
        lines = [
            f"检索模式：{report['mode']}",
            f"Embedding 可用：{report['embedding_available']}",
            f"向量文档数：{report['vector_count']}",
            f"关键词候选：{len(report['keyword'])}",
            f"阈值：{report['threshold']}",
        ]
        lines += [f"关键词 {title}: {score}" for title, score in report["keyword"][:10]]
        lines += [
            f"向量 {title}: {score:.3f}" for title, score in report["vector"][:10]
        ]
        lines += [
            f"最终 《{x.document.title}》: {x.score:.3f}" for x in report["selected"]
        ]
        return "\n".join(lines)

    async def rebuild_embeddings(self) -> int:
        count = 0
        for row in self.index.all_documents():
            vector = await self._embed(row["title"] + "\n" + row["body"])
            if vector:
                self.index.upsert(self._doc(row), vector)
                count += 1
        return count
