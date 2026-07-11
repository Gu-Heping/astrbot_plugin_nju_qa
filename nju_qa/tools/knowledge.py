from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from ..agent import SourceTracker
from ..knowledge_tools import search_knowledge_base
from ..retriever import HybridRetriever


@dataclass
class SearchKnowledgeBaseTool(FunctionTool):
    """Search synchronized NJU documents and return only traceable source material."""

    name: str = "search_knowledge_base"
    description: str = (
        "检索南京大学知识库。涉及南京大学具体事实、政策、流程、时间、地点、"
        "联系方式或课程要求时必须先调用此工具。"
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的完整问题或关键词"}
            },
            "required": ["query"],
        }
    )
    retriever: HybridRetriever | None = None
    tracker: SourceTracker | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> str:
        """Current AstrBot tool-loop entry point."""

        return await self._search(str(kwargs.get("query", "")))

    async def run(self, event: AstrMessageEvent, query: str) -> str:
        """Compatibility entry point for AstrBot versions using ``run``."""

        return await self._search(query)

    async def _search(self, query: str) -> str:
        if self.retriever is None or self.tracker is None:
            return "知识库工具未初始化。"
        return await search_knowledge_base(self.retriever, self.tracker, query)
