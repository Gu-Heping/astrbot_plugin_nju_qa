from __future__ import annotations
from collections.abc import Awaitable, Callable
from .prompts import SYSTEM_PROMPT, build_prompt
from .retriever import HybridRetriever

NO_ANSWER = "知识库中暂未找到可靠答案"


class AnswerService:
    def __init__(
        self, retriever: HybridRetriever, llm: Callable[[str, str], Awaitable[str]]
    ):
        self.retriever, self.llm = retriever, llm

    async def answer(self, question: str) -> str:
        sources = await self.retriever.search(question)
        if not sources:
            return NO_ANSWER
        try:
            text = await self.llm(build_prompt(question, sources), SYSTEM_PROMPT)
        except Exception:
            return NO_ANSWER
        return text.strip() + "\n\n" + self._citations(sources)

    async def source_results(self, query: str):
        return await self.retriever.search(query)

    def format_source_results(self, sources) -> str:
        return self._citations(sources) if sources else NO_ANSWER

    def _citations(self, sources) -> str:
        return "参考来源：\n" + "\n".join(
            f"{i}. 《{s.document.title}》：{s.document.url}"
            for i, s in enumerate(sources, 1)
        )
