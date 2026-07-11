"""Framework-independent implementation behind the Agent knowledge tool."""

from __future__ import annotations

from .agent import SourceTracker
from .retriever import HybridRetriever


async def search_knowledge_base(
    retriever: HybridRetriever, tracker: SourceTracker, query: str
) -> str:
    """Search and record source evidence returned to the Agent."""

    if not retriever.index.all_documents():
        return (
            "知识库尚未同步；请提示用户联系管理员先执行 /nju_sync，不能编造校园事实。"
        )
    results = await retriever.search(query)
    if not results:
        return "知识库中暂未找到可靠答案；请明确告诉用户没有可靠资料，不要编造。"
    tracker.add(results)
    blocks = []
    for result in results:
        document = result.document
        blocks.append(
            f"[{result.source_id}]《{document.title}》\n"
            f"更新时间：{document.updated_at}\n"
            f"原文链接：{document.url}\n"
            f"内容：{document.body[:3500]}"
        )
    return "\n\n".join(blocks)
