"""Thin adapter around AstrBot's native tool-loop agent."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from .models import SearchResult
from .prompts import AGENT_SYSTEM_PROMPT


NO_PROVIDER = "当前未配置 LLM 服务。请联系管理员配置后再试。"
AGENT_ERROR = "当前无法调用 LLM 服务，请稍后重试。"


class SourceTracker:
    """Collects only sources that an Agent tool actually returned this turn."""

    def __init__(self) -> None:
        self.sources: list[SearchResult] = []

    def reset(self) -> None:
        self.sources.clear()

    def add(self, results: list[SearchResult]) -> None:
        known = {item.document.yuque_id for item in self.sources}
        for item in results:
            if item.document.yuque_id not in known:
                self.sources.append(item)
                known.add(item.document.yuque_id)


def append_verified_citations(text: str, sources: list[SearchResult]) -> str:
    """Drop a model-generated source section and render only tracked sources."""

    text = re.split(r"\n\s*(?:参考来源|来源)\s*[:：]", text, maxsplit=1)[0].strip()
    allowed_urls = {result.document.url for result in sources}
    text = re.sub(
        r"https?://[^\s<>()]+",
        lambda match: (
            match.group(0) if match.group(0) in allowed_urls else "[未验证链接已移除]"
        ),
        text,
    )
    if not sources:
        return text
    citations = "\n".join(
        f"{number}. 《{result.document.title}》：{result.document.url}"
        for number, result in enumerate(sources, 1)
    )
    return f"{text}\n\n参考来源：\n{citations}"


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
        return append_verified_citations(
            str(getattr(response, "completion_text", "")).strip(), tracker.sources
        )

    async def _run_tool_loop(
        self, *, tracker: SourceTracker, **kwargs: object
    ) -> object:
        tools = self.tools(tracker)
        if self._tool_loop is not None:
            return await self._tool_loop(tools=tools, **kwargs)

        # Imported lazily so core tests do not require a full AstrBot installation.
        from astrbot.core.agent.tool import ToolSet

        return await self.context.tool_loop_agent(
            tools=ToolSet(tools), max_steps=8, tool_call_timeout=60, **kwargs
        )
