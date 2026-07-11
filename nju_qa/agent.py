"""Thin adapter around AstrBot's native tool-loop agent."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from .models import SearchResult
from .prompts import AGENT_SYSTEM_PROMPT


NO_PROVIDER = "当前未配置 LLM 服务。请联系管理员配置后再试。"
AGENT_ERROR = "当前无法调用 LLM 服务，请稍后重试。"
NO_EVIDENCE = "知识库中暂未找到可靠资料，我不能据此给出南京大学的具体结论。"


class SourceTracker:
    """Collects only sources that an Agent tool actually returned this turn."""

    def __init__(self) -> None:
        self.sources: list[SearchResult] = []
        self.read_sources: set[str] = set()
        self.verified_urls: set[str] = set()

    def reset(self) -> None:
        self.sources.clear()
        self.read_sources.clear()
        self.verified_urls.clear()

    def record_read_content(self, content: str) -> None:
        self.verified_urls.update(re.findall(r"https?://[^\s<>()，。；：]+", content))

    def add(self, results: list[SearchResult]) -> None:
        known = {item.document.yuque_id for item in self.sources}
        for item in results:
            if item.document.yuque_id not in known:
                self.sources.append(item)
                known.add(item.document.yuque_id)


def append_verified_citations(
    text: str, sources: list[SearchResult], verified_urls: set[str] | None = None
) -> str:
    """Drop a model-generated source section and render only tracked sources."""

    text = re.split(r"\n\s*(?:参考来源|来源)\s*[:：]", text, maxsplit=1)[0].strip()
    allowed_urls = {result.document.url for result in sources}
    allowed_urls.update(verified_urls or set())
    text = re.sub(
        r"https?://[^\s<>()，。；：]+",
        lambda match: match.group(0) if match.group(0) in allowed_urls else "",
        text,
    )
    if not sources:
        return text
    citations = "\n".join(
        f"{number}. 《{result.document.title}》：{result.document.url}"
        for number, result in enumerate(sources, 1)
    )
    return f"{text}\n\n参考来源：\n{citations}"


def requires_campus_evidence(prompt: str) -> bool:
    """Conservative multi-signal guard; the Agent still chooses which tools to use."""
    topics = (
        "新生",
        "校园",
        "南大",
        "南京大学",
        "学分",
        "课程",
        "教务",
        "门户",
        "网站",
        "补办",
        "转专业",
        "校区",
        "住宿",
        "奖助",
        "考试",
        "流程",
    )
    factual = (
        "哪里",
        "怎么",
        "多少",
        "什么",
        "要求",
        "需要",
        "时间",
        "地址",
        "网站",
        "办理",
        "吗",
        "？",
        "?",
    )
    return sum(token in prompt for token in topics) >= 2 or (
        any(token in prompt for token in topics)
        and any(token in prompt for token in factual)
    )


ToolFactory = Callable[[SourceTracker], list[object]]
ToolLoop = Callable[..., Awaitable[object]]


class NjuQaAgent:
    """Uses AstrBot's current provider and tool-loop instead of intent heuristics."""

    def __init__(
        self, context: object, tools: ToolFactory, tool_loop: ToolLoop | None = None
    ):
        self.context = context
        self.tools = tools
        self._tool_loop = tool_loop

    async def answer(self, event: object, prompt: str) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        if not provider_id:
            return NO_PROVIDER

        tracker = SourceTracker()
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=AGENT_SYSTEM_PROMPT,
                tracker=tracker,
            )
        except Exception:
            return AGENT_ERROR
        text = str(getattr(response, "completion_text", "")).strip()

        if not requires_campus_evidence(prompt):
            return append_verified_citations(text, tracker.sources, tracker.verified_urls)

        # For campus-factual questions, require reliable sources.
        reliable_sources = [s for s in tracker.sources if s.reliable]
        if not reliable_sources:
            return NO_EVIDENCE

        # Ground the answer in the most relevant chunk snippets.
        self._record_selected_documents(tracker)
        response = await self._run_tool_loop(
            event=event,
            chat_provider_id=provider_id,
            prompt=self._grounded_prompt(prompt, tracker),
            system_prompt=AGENT_SYSTEM_PROMPT,
            tracker=tracker,
        )
        text = str(getattr(response, "completion_text", "")).strip()
        if not tracker.read_sources:
            return NO_EVIDENCE
        return append_verified_citations(text, tracker.sources, tracker.verified_urls)

    @staticmethod
    def _record_selected_documents(tracker: SourceTracker) -> None:
        """Use chunk snippets for grounding instead of full document bodies."""
        for source in tracker.sources[:3]:
            tracker.read_sources.add(
                str(source.document.path or source.document.yuque_id)
            )
            if source.chunk is not None:
                tracker.record_read_content(source.chunk.content_snippet)
            else:
                tracker.record_read_content(source.document.body)

    @staticmethod
    def _grounded_prompt(question: str, tracker: SourceTracker) -> str:
        materials = "\n\n".join(
            f"[已读材料 {i}]《{source.document.title}》\n{source.chunk.content_snippet[:6000] if source.chunk else source.document.body[:6000]}"
            for i, source in enumerate(tracker.sources[:3], 1)
        )
        return f"""请回答原问题：{question}

只能根据以下已读的知识库正文作答。正文中没有提到的具体事项，必须明确说“知识库中暂未找到可靠资料”，绝不能补充一般经验、网站、流程或联系方式。不要自行输出链接或来源列表。

{materials}"""

    async def _run_tool_loop(
        self, *, tracker: SourceTracker, **kwargs: object
    ) -> object:
        tools = self.tools(tracker)
        if self._tool_loop is not None:
            return await self._tool_loop(tools=tools, **kwargs)

        # Imported lazily so core tests do not require a full AstrBot installation.
        from astrbot.core.agent.tool import ToolSet

        return await self.context.tool_loop_agent(
            tools=ToolSet(tools), max_steps=12, tool_call_timeout=60, **kwargs
        )
