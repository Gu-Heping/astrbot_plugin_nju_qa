"""Framework-independent implementation behind the Agent knowledge tool."""

from __future__ import annotations

from .agent import SourceTracker
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
        "reliable": True,
        "candidates": [
            {
                "source_id": r.source_id,
                "title": r.document.title,
                "author": "",
                "book_name": r.document.repository,
                "content_snippet": r.document.body[:1200],
                "file_path": str(r.document.path or ""),
                "yuque_id": r.document.yuque_id,
                "slug": r.document.slug,
                "source_url": r.document.url,
                "score": r.score,
                "score_type": "combined_similarity",
                "retrieval_method": "hybrid",
                "reliable": True,
            }
            for r in results
        ],
    }
