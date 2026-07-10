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
        merged: dict[str, float] = {}
        for row, score in self.index.keyword(query, self.config.retrieval_top_k * 2):
            merged[row["yuque_id"]] = max(
                merged.get(row["yuque_id"], 0), min(1, score / 3)
            )
        vector = await self._embed(query)
        if vector:
            for row, score in self.index.vector(
                vector, self.config.retrieval_top_k * 2
            ):
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
        return selected

    async def rebuild_embeddings(self) -> int:
        count = 0
        for row in self.index.all_documents():
            vector = await self._embed(row["title"] + "\n" + row["body"])
            if vector:
                self.index.upsert(self._doc(row), vector)
                count += 1
        return count
