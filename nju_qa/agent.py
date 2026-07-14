"""Thin adapter around AstrBot's native tool-loop agent."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .doc_utils import read_document_content
from .entities import QueryEvidenceMode, classify_evidence_mode
from .evidence import (
    grep_hits_to_search_results,
    merge_search_results,
    select_grounding_sources,
)
from .models import ChunkResult, Document, SearchResult
from .prompts import AGENT_SYSTEM_PROMPT
from .retrieval_executor import RetrievalExecutor
from .retrieval_plan import (
    CoverageResult,
    CoverageStatus,
    build_retrieval_plan,
    check_coverage,
    format_coverage_report,
    format_plan,
)


NO_PROVIDER = "当前未配置 LLM 服务。请联系管理员配置后再试。"
AGENT_ERROR = "当前无法调用 LLM 服务，请稍后重试。"
NO_EVIDENCE = "知识库中暂未找到可靠资料，我不能据此给出南京大学的具体结论。"

_MAX_GROUNDING_SOURCES = 7
_NO_EVIDENCE_MARKERS = ("知识库中暂未找到可靠资料", "知识库中暂未找到可靠答案")


class SourceTracker:
    """Collects only sources that an Agent tool actually returned this turn."""

    def __init__(self) -> None:
        self.sources: list[SearchResult] = []
        self.selected_sources: list[SearchResult] = []
        self.read_sources: set[str] = set()
        self.verified_urls: set[str] = set()

    def reset(self) -> None:
        self.sources.clear()
        self.selected_sources.clear()
        self.read_sources.clear()
        self.verified_urls.clear()

    def record_read_content(self, content: str) -> None:
        self.verified_urls.update(re.findall(r"https?://[^\s<>()，。；：]+", content))

    def add(self, results: list[SearchResult]) -> None:
        """Add or merge retrieval results into the unified evidence list."""
        by_id: dict[str, int] = {
            item.document.yuque_id: i for i, item in enumerate(self.sources)
        }
        for item in results:
            if not item.document.yuque_id:
                # Documents without a stable id are appended as-is (rare).
                self.sources.append(item)
                continue
            idx = by_id.get(item.document.yuque_id)
            if idx is None:
                self.sources.append(item)
                by_id[item.document.yuque_id] = len(self.sources) - 1
            else:
                merged = merge_search_results(self.sources[idx], item)
                self.sources[idx] = merged
                by_id[item.document.yuque_id] = idx
            content = (
                item.chunk.content_snippet
                if item.chunk
                else item.document.body
            )
            if content:
                self.record_read_content(content)

    def add_grep_hits(self, hits: list[dict], query_terms: list[str]) -> None:
        """Convert grep hits into unified evidence and register them."""
        self.add(grep_hits_to_search_results(hits, query_terms))

    def add_read_document(self, document: Document, content: str) -> None:
        """Mark a document as read and ensure it exists in the evidence list."""
        key = str(document.path or document.yuque_id)
        self.read_sources.add(key)
        self.record_read_content(content)
        if not document.yuque_id:
            return
        # Merge with any existing evidence for the same document.
        existing_idx = next(
            (
                i
                for i, s in enumerate(self.sources)
                if s.document.yuque_id == document.yuque_id
            ),
            None,
        )
        if existing_idx is None:
            # A document that is read without prior retrieval evidence is
            # registered, but it does not automatically become reliable just
            # because it was opened.
            self.add(
                [
                    SearchResult(
                        source_id=f"R{len(self.sources) + 1}",
                        document=document,
                        score=1.0,
                        chunk=None,
                        retrieval_methods=("read",),
                        reliable=False,
                    )
                ]
            )
        else:
            existing = self.sources[existing_idx]
            merged_document = document
            if len(existing.document.body) > len(document.body):
                merged_document = existing.document
            read_chunk = (
                existing.chunk
                or ChunkResult(
                    chunk_id=f"read:{document.yuque_id}",
                    document_id=document.yuque_id,
                    title=document.title,
                    content_snippet=content[:2400],
                    source_url=document.url,
                    final_score=existing.score,
                    retrieval_methods=("read",),
                    reliable=existing.reliable,
                    file_path=str(document.path or ""),
                    slug=document.slug,
                    namespace=document.namespace,
                )
            )
            self.sources[existing_idx] = SearchResult(
                source_id=existing.source_id,
                document=merged_document,
                score=existing.score,
                chunk=read_chunk,
                keyword_score=existing.keyword_score,
                retrieval_methods=tuple(
                    dict.fromkeys([*existing.retrieval_methods, "read"])
                ),
                reliable=existing.reliable,
            )

    @property
    def reliable_count(self) -> int:
        return sum(1 for s in self.sources if s.reliable)

    @property
    def matched_count(self) -> int:
        return len(self.sources)

    @property
    def read_count(self) -> int:
        return len(self.read_sources)


def _is_pure_no_evidence(text: str) -> bool:
    """Return True when the answer contains no substantive information."""
    cleaned = text
    for marker in _NO_EVIDENCE_MARKERS:
        cleaned = cleaned.replace(marker, "")
    # Drop punctuation and whitespace; if nothing meaningful remains, it is a pure
    # no-evidence answer.
    cleaned = re.sub(r"[^\w一-鿿]", "", cleaned).strip()
    return not cleaned


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
    # Do not append a source list to an answer that contains no substantive
    # information; otherwise keep the sources so the user can inspect the original
    # documents.
    if _is_pure_no_evidence(text) or not sources:
        logger.info("NJU agent: suppressing citations for no-evidence answer")
        return text
    # Limit citations to the most relevant sources to avoid overwhelming the user.
    top_sources = sources[:5]
    citations = "\n".join(
        f"{number}. 《{result.document.title}》：{result.document.url}"
        for number, result in enumerate(top_sources, 1)
    )
    logger.info("NJU agent: appending %d citations", len(top_sources))
    return f"{text}\n\n参考来源：\n{citations}"


ToolFactory = Callable[[SourceTracker], list[object]]
ToolLoop = Callable[..., Awaitable[object]]


class NjuQaAgent:
    """Uses AstrBot's native tool-loop agent with optional full-document grounding."""

    def __init__(
        self,
        context: object,
        tools: ToolFactory,
        tool_loop: ToolLoop | None = None,
        docs_root: Path | None = None,
        index: Any = None,
    ):
        self.context = context
        self.tools = tools
        self._tool_loop = tool_loop
        self.docs_root = docs_root
        self.index = index

    async def answer(self, event: object, prompt: str) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        logger.info("NJU agent start: provider=%s prompt=%r", provider_id, prompt)
        if not provider_id:
            logger.warning("NJU agent: no chat provider configured")
            return NO_PROVIDER

        tracker = SourceTracker()
        rows = self.index.all_documents() if self.index is not None else []
        evidence_mode = classify_evidence_mode(prompt)
        logger.info("NJU agent evidence mode: %s", evidence_mode.value)

        if evidence_mode == QueryEvidenceMode.CAMPUS_FACTUAL:
            if rows and self.retriever is not None:
                return await self._answer_campus_factual(event, prompt, tracker, rows)
            # Fallback for environments without a configured retriever (e.g. some
            # unit tests): let the LLM use tools and apply the same evidence gate.
            return await self._answer_campus_factual_fallback(
                event, prompt, tracker, rows
            )

        # Non-factual small talk: use the LLM directly.
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=AGENT_SYSTEM_PROMPT,
                tracker=tracker,
            )
        except Exception:
            logger.exception("NJU agent tool loop failed")
            return AGENT_ERROR
        text = str(getattr(response, "completion_text", "")).strip()
        result = append_verified_citations(
            text, tracker.sources, tracker.verified_urls
        )
        logger.info("NJU agent direct answer: length=%d", len(result))
        return result

    async def _answer_campus_factual(
        self,
        event: object,
        prompt: str,
        tracker: SourceTracker,
        rows: list[Any],
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        plan = build_retrieval_plan(prompt, rows)
        logger.info("NJU agent plan: %d subquestions", len(plan))

        executor = RetrievalExecutor(self.retriever, self.index, self.docs_root)
        exec_result = await executor.execute(plan, tracker)
        logger.info(
            "NJU agent executor: sources=%d reliable=%d zero_entity_hits=%d",
            len(tracker.sources),
            tracker.reliable_count,
            len(exec_result.zero_entity_hits),
        )

        reliable_sources = [s for s in tracker.sources if s.reliable]
        if not reliable_sources:
            logger.info("NJU agent: no reliable sources for campus question")
            return NO_EVIDENCE

        answer_mode = self._classify_answer_mode(exec_result.coverage)
        if answer_mode == "UNSUPPORTED":
            logger.info("NJU agent: no direct or background coverage")
            return NO_EVIDENCE

        supporting_sources = self._collect_supporting_sources(exec_result.coverage)
        tracker.selected_sources = select_grounding_sources(
            supporting_sources, max_sources=_MAX_GROUNDING_SOURCES
        )
        self._record_selected_documents(tracker)
        if not tracker.selected_sources:
            logger.info("NJU agent: no selected grounding sources")
            return NO_EVIDENCE

        logger.info(
            "NJU agent grounding: mode=%s selected=%d sources=%d reliable=%d",
            answer_mode,
            len(tracker.selected_sources),
            len(tracker.sources),
            tracker.reliable_count,
        )

        system_prompt = f"{AGENT_SYSTEM_PROMPT}\n\n{format_plan(plan)}"
        grounded_prompt = self._grounded_prompt(
            prompt, tracker, exec_result.coverage, answer_mode
        )

        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=grounded_prompt,
                system_prompt=system_prompt,
                tracker=tracker,
            )
        except Exception:
            logger.exception("NJU agent grounded tool loop failed")
            return AGENT_ERROR
        text = str(getattr(response, "completion_text", "")).strip()
        if not tracker.read_sources:
            logger.info("NJU agent: no documents were read during grounding")
            return NO_EVIDENCE
        result = append_verified_citations(
            text, tracker.selected_sources, tracker.verified_urls
        )
        logger.info(
            "NJU agent grounded answer: length=%d selected=%d reliable=%d",
            len(result),
            len(tracker.selected_sources),
            tracker.reliable_count,
        )
        return result

    async def _answer_campus_factual_fallback(
        self,
        event: object,
        prompt: str,
        tracker: SourceTracker,
        rows: list[Any],
    ) -> str:
        """Tool-loop fallback when no retriever/index is available."""
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        plan = None
        system_prompt = AGENT_SYSTEM_PROMPT
        if rows:
            plan = build_retrieval_plan(prompt, rows)
            system_prompt = f"{AGENT_SYSTEM_PROMPT}\n\n{format_plan(plan)}"

        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt,
                tracker=tracker,
            )
        except Exception:
            logger.exception("NJU agent fallback tool loop failed")
            return AGENT_ERROR
        text = str(getattr(response, "completion_text", "")).strip()
        logger.info(
            "NJU agent fallback first response: length=%d sources=%d reliable=%d",
            len(text),
            len(tracker.sources),
            tracker.reliable_count,
        )

        reliable_sources = [s for s in tracker.sources if s.reliable]
        if not reliable_sources:
            logger.info("NJU agent fallback: no reliable sources")
            return NO_EVIDENCE

        # Entity-specific subquestions still require DIRECT evidence.
        if plan is not None:
            coverage = check_coverage(plan, tracker.sources)
            for cov in coverage:
                if cov.need.entity_terms and cov.status != CoverageStatus.DIRECT:
                    logger.info(
                        "NJU agent fallback: subquestion %r lacks direct evidence",
                        cov.need.question,
                    )
                    return NO_EVIDENCE

        tracker.selected_sources = select_grounding_sources(
            reliable_sources, max_sources=_MAX_GROUNDING_SOURCES
        )
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
            logger.info("NJU agent fallback: no documents read during grounding")
            return NO_EVIDENCE
        return append_verified_citations(
            text, tracker.selected_sources, tracker.verified_urls
        )

    @staticmethod
    def _classify_answer_mode(coverage: list[CoverageResult]) -> str:
        """Return DIRECT_ANSWER, PARTIAL_ANSWER, BACKGROUND_ONLY or UNSUPPORTED."""
        has_direct = any(c.status == CoverageStatus.DIRECT for c in coverage)
        has_background = any(c.status == CoverageStatus.BACKGROUND for c in coverage)
        has_unsupported = any(
            c.status == CoverageStatus.UNSUPPORTED for c in coverage
        )
        if has_direct:
            return "PARTIAL_ANSWER" if has_unsupported else "DIRECT_ANSWER"
        if has_background:
            return "BACKGROUND_ONLY"
        return "UNSUPPORTED"

    @staticmethod
    def _collect_supporting_sources(
        coverage: list[CoverageResult],
    ) -> list[SearchResult]:
        """Gather reliable sources that back DIRECT or BACKGROUND coverage."""
        supporting: list[SearchResult] = []
        seen: set[str] = set()
        for cov in coverage:
            if cov.status not in (CoverageStatus.DIRECT, CoverageStatus.BACKGROUND):
                continue
            for source in cov.sources:
                if not getattr(source, "reliable", False):
                    continue
                key = source.document.yuque_id or source.document.url or source.document.title
                if key in seen:
                    continue
                seen.add(key)
                supporting.append(source)
        return supporting

    @property
    def retriever(self) -> Any:
        """Best-effort retriever discovery from the configured tools.

        The executor needs a :class:`HybridRetriever`; when the Agent was built
        without one we derive it from ``search_knowledge_base`` tools.
        """
        # Plugins instantiate the Agent before the retriever is attached, so the
        # retriever is discovered lazily from the tool factory.
        if hasattr(self, "_retriever") and self._retriever is not None:
            return self._retriever
        # Build tools once (without a real tracker) to inspect them.
        try:
            dummy_tracker = SourceTracker()
            tools = self.tools(dummy_tracker)
            for tool in tools:
                if getattr(tool, "name", "") == "search_knowledge_base":
                    retriever = getattr(tool, "retriever", None)
                    if retriever is not None:
                        self._retriever = retriever
                        return retriever
        except Exception:
            logger.debug("NJU agent: could not discover retriever from tools")
        return None

    def _read_source_body(self, source: SearchResult, limit: int = 8000) -> str:
        """Return full document body when docs_root is configured, else chunk snippet."""
        if self.docs_root is None or source.document.path is None:
            return (
                source.chunk.content_snippet[:limit]
                if source.chunk
                else source.document.body[:limit]
            )
        try:
            result = read_document_content(
                self.docs_root, str(source.document.path), offset=0, limit=limit
            )
            return result["content"]
        except ValueError:
            return (
                source.chunk.content_snippet[:limit]
                if source.chunk
                else source.document.body[:limit]
            )

    def _record_selected_documents(self, tracker: SourceTracker) -> None:
        """Mark the already-selected grounding sources as read and extract URLs."""
        for source in tracker.selected_sources:
            tracker.read_sources.add(
                str(source.document.path or source.document.yuque_id)
            )
            tracker.record_read_content(self._read_source_body(source))

    def _grounded_prompt(
        self,
        question: str,
        tracker: SourceTracker,
        coverage: list[CoverageResult] | None = None,
        answer_mode: str | None = None,
    ) -> str:
        sources = tracker.selected_sources
        materials = "\n\n".join(
            f"[已读材料 {i}]《{source.document.title}》（{source.document.url}）\n{self._read_source_body(source, limit=8000)}"
            for i, source in enumerate(sources, 1)
        )

        coverage_text = ""
        if coverage:
            coverage_text = "\n" + format_coverage_report(coverage)

        mode_instructions = {
            "DIRECT_ANSWER": "所有子问题都有 DIRECT 证据，请直接整理成条理清晰的答案。",
            "PARTIAL_ANSWER": (
                "部分子问题只有 DIRECT 证据，其余子问题没有直接证据。"
                "请明确回答有 DIRECT 证据的部分；对于没有直接证据的子问题，"
                "直接说明“知识库中暂未找到可靠资料”，不要编造具体事实。"
            ),
            "BACKGROUND_ONLY": (
                "没有针对具体实体的 DIRECT 证据，只有通用背景材料。"
                "请只根据材料给出通用背景说明，并提醒用户该问题未在知识库中找到针对具体实体的可靠资料。"
            ),
        }.get(answer_mode, "")

        return f"""请回答原问题：{question}

{mode_instructions}{coverage_text}

我已为你读取了以下最相关的知识库文档（或片段）。请严格根据这些材料作答：
- 只回答用户实际询问的事项；用户未指定需要逐校区穷举时，不要主动列出材料未覆盖的校区；材料未提及的校区、流程或建议直接略过；不得自行建议前往其他校区办理，除非材料明确支持。
- 地点、时间、费用、校区等条件必须与对应材料绑定；不同来源或校区信息不一致时，不得合并成一个统一结论，可分别说明或提示资料存在版本差异。
- 若材料为历史归档或包含具体年份/截止日期，不要把该截止日期当作当前截止日期；可说明历年操作路径，但提醒以本年度通知为准。
- 材料中明确提到的具体事项，直接整理成条理清晰的答案；材料中没有提到的具体事项直接略过，不要单独列出“未找到”或“未提及”的事项清单。
- 如果某个材料明显不足，可以调用 read_doc(file_path) 读取完整文档，但只读取已列出的文档；不要自行输出链接或来源列表，系统会自动附加。

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
