"""Framework-independent implementation behind the Agent knowledge tool."""

from __future__ import annotations

from .agent import SourceTracker
from .models import ChunkResult
from .retriever import HybridRetriever


async def search_knowledge_base(
    retriever: HybridRetriever, tracker: SourceTracker, query: str
) -> dict:
    """Search and record source evidence returned to the Agent."""

    if not retriever.index.all_documents():
        return {
            "reliable": False,
            "reason": "知识库尚未同步；请管理员先执行 /nju_sync。",
            "candidates": [],
        }
    results = await retriever.search(query)
    if not results:
        return {
            "reliable": False,
            "reason": "知识库中暂未找到可靠答案。",
            "candidates": [],
        }
    tracker.add(results)
    return {
        "reliable": all(r.reliable for r in results),
        "candidates": [_result_to_dict(r.chunk, r) for r in results],
    }


def _result_to_dict(chunk: ChunkResult | None, result) -> dict:
    if chunk is None:
        return {
            "source_id": result.source_id,
            "title": result.document.title,
            "author": "",
            "book_name": result.document.repository,
            "content_snippet": result.document.body[:1200],
            "file_path": str(result.document.path or ""),
            "yuque_id": result.document.yuque_id,
            "slug": result.document.slug,
            "source_url": result.document.url,
            "score": result.score,
            "score_type": "combined_similarity",
            "retrieval_method": "hybrid",
            "reliable": result.reliable,
        }
    return {
        "source_id": result.source_id,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "author": "",
        "book_name": result.document.repository,
        "content_snippet": chunk.content_snippet[:1200],
        "file_path": chunk.file_path or str(result.document.path or ""),
        "yuque_id": result.document.yuque_id,
        "slug": chunk.slug or result.document.slug,
        "source_url": chunk.source_url or result.document.url,
        "score": chunk.final_score,
        "vector_raw_score": chunk.vector_raw_score,
        "vector_score_type": chunk.vector_score_type,
        "vector_relevance": chunk.vector_relevance,
        "keyword_score": chunk.keyword_score,
        "final_score": chunk.final_score,
        "retrieval_methods": list(chunk.retrieval_methods),
        "reliable": chunk.reliable,
    }
