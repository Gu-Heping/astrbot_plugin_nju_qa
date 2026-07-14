"""Evidence-first two-stage Agent for NJU QA.

The Agent splits every factual question into two phases:

1. Research phase: the model may use search/navigation/reading tools to locate
   and read concrete evidence.  The natural-language text produced by this
   phase is ignored; only the evidence that was actually read is retained.

2. Answer phase: the model receives the original question and the collected
   evidence excerpts, and must answer using only those excerpts.  Every factual
   claim must be marked with an internal evidence id ``[E#]``.  Citations are
   rendered only for the excerpts the model actually used.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .doc_utils import read_document_content
from .evidence import (
    EvidenceExcerpt,
    _NO_ANSWER_MARKERS,
    evidence_excerpt_from_read,
)
from .models import SearchResult
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT,
)


NO_PROVIDER = "当前未配置 LLM 服务。请联系管理员配置后再试。"
AGENT_ERROR = "当前无法调用 LLM 服务，请稍后重试。"
NO_EVIDENCE = "知识库中暂未找到可靠资料，我不能据此给出南京大学的具体结论。"
SAFE_FAILURE = "当前无法根据知识库给出可靠回答，请稍后重试或换个方式提问。"

_MAX_EVIDENCE = 7
_MAX_RESEARCH_STEPS = 12
_MAX_ANSWER_STEPS = 3
_NO_EVIDENCE_MARKERS = ("知识库中暂未找到可靠资料", "知识库中暂未找到可靠答案")

_SMALL_TALK_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(你好|您好|嗨|哈喽|hello|hi|hey|在吗|在不在|早上好|下午好|晚上好)([！!。，,？?\s]|$)", re.IGNORECASE),
    re.compile(r"^(谢谢|多谢|感谢|拜拜|再见|bye|goodbye)([！!。，,？?\s]|$)", re.IGNORECASE),
    re.compile(r"^(你?是谁|你叫什么|你叫什么名字|介绍一下你|自我介绍一下)"),
    re.compile(r"^(你?能做什么|你?会什么|你?有什么功能|帮助|help|怎么用)"),
    re.compile(r"^(好的|行|可以|ok|okay|知道了|明白)([！!。，,？?\s]|$)", re.IGNORECASE),
]


def _is_small_talk(prompt: str) -> bool:
    """Return True only when the whole message is a pure conversational opener."""
    text = prompt.strip()
    if not text:
        return True
    for pattern in _SMALL_TALK_PATTERNS:
        match = pattern.search(text)
        if match:
            # Anything beyond the greeting (except trailing punctuation/space)
            # means the message also contains a factual question.
            tail = text[match.end():].strip(" \t\n\r！!。，,、；：:？?\"'")
            if not tail:
                return True
    return False


def _strip_unverified_urls(text: str, allowed_urls: set[str]) -> str:
    """Remove URLs not present in ``allowed_urls`` to prevent hallucinated links."""

    def replace(match: re.Match) -> str:
        url = match.group(0)
        # Allow exact matches or URLs that share a prefix with an allowed URL.
        if url in allowed_urls:
            return url
        for allowed in allowed_urls:
            if url.startswith(allowed) or allowed.startswith(url):
                return url
        return ""

    return re.sub(r"https?://[^\s<>()，。；：）]+", replace, text)


def _extract_used_evidence_ids(text: str) -> list[str]:
    """Return ordered evidence ids used by the model, e.g. ['E1','E3']."""
    return list(dict.fromkeys(f"E{m}" for m in re.findall(r"\[E(\d+)]", text)))


def _is_pure_no_evidence(text: str) -> bool:
    """Return True when the answer contains no substantive information."""
    cleaned = text
    for marker in _NO_EVIDENCE_MARKERS:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"[^\w一-鿿]", "", cleaned).strip()
    return not cleaned


@dataclass
class SourceTracker:
    """Tracks candidate sources and concrete evidence excerpts.

    * candidate_sources  -- search/grep results used to locate documents.
    * evidence_excerpts  -- actual text read by the model (the only thing that
                            may ground a factual answer).
    * selected_excerpts  -- excerpts chosen for the final answer prompt.
    """

    candidate_sources: list[SearchResult] = field(default_factory=list)
    evidence_excerpts: list[EvidenceExcerpt] = field(default_factory=list)
    selected_excerpts: list[EvidenceExcerpt] = field(default_factory=list)
    read_sources: set[str] = field(default_factory=set)
    verified_urls: set[str] = field(default_factory=set)
    diagnostics: bool = False

    def reset(self) -> None:
        self.candidate_sources.clear()
        self.evidence_excerpts.clear()
        self.selected_excerpts.clear()
        self.read_sources.clear()
        self.verified_urls.clear()

    def record_urls(self, content: str) -> None:
        self.verified_urls.update(re.findall(r"https?://[^\s<>()，。；：]+", content))

    def add_candidates(self, results: list[SearchResult]) -> None:
        """Register search results as candidates, not as final evidence."""
        by_id: dict[str, int] = {
            item.document.yuque_id: i
            for i, item in enumerate(self.candidate_sources)
            if item.document.yuque_id
        }
        for item in results:
            content = (
                item.chunk.content_snippet if item.chunk else item.document.body
            )
            if content:
                self.record_urls(content)
            if not item.document.yuque_id:
                self.candidate_sources.append(item)
                continue
            idx = by_id.get(item.document.yuque_id)
            if idx is None:
                self.candidate_sources.append(item)
                by_id[item.document.yuque_id] = len(self.candidate_sources) - 1
            else:
                existing = self.candidate_sources[idx]
                if item.score > existing.score:
                    self.candidate_sources[idx] = item
                    by_id[item.document.yuque_id] = idx

    def add_grep_hits(self, hits: list[dict], query_terms: list[str]) -> None:
        """Compatibility shim: grep hits are candidate evidence only."""
        from .evidence import grep_hits_to_search_results

        self.add_candidates(grep_hits_to_search_results(hits, query_terms))

    def add_read_document(
        self,
        document,
        content: str,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> None:
        """Record a document read as concrete evidence."""
        key = str(document.path or document.yuque_id)
        self.read_sources.add(key)
        self.record_urls(content)
        self.add_evidence(
            evidence_excerpt_from_read(
                document,
                content[:2400],
                line_start=line_start,
                line_end=line_end,
            )
        )

    def add_evidence(self, excerpt: EvidenceExcerpt) -> EvidenceExcerpt:
        """Append an evidence excerpt and assign it an internal id."""
        if not excerpt.evidence_id:
            excerpt.evidence_id = f"E{len(self.evidence_excerpts) + 1}"
        self.evidence_excerpts.append(excerpt)
        self.read_sources.add(excerpt.file_path or excerpt.document_id)
        self.record_urls(excerpt.content)
        return excerpt

    def has_negative_evidence(self, question: str) -> bool:
        """Return True when a highly similar QA entry explicitly says no answer."""
        norm_q = question.casefold()
        for excerpt in self.evidence_excerpts:
            if excerpt.qa_status != "no_answer":
                continue
            # Require the negative QA to mention the question subject.
            if any(marker in excerpt.content.casefold() for marker in _NO_ANSWER_MARKERS):
                if self._overlap_score(norm_q, excerpt.content) >= 0.3:
                    return True
        return False

    @staticmethod
    def _overlap_score(query_norm: str, text: str) -> float:
        query_chars = set(query_norm)
        text_chars = set(text.casefold())
        if not query_chars:
            return 0.0
        return len(query_chars & text_chars) / len(query_chars)

    @property
    def read_count(self) -> int:
        return len(self.read_sources)


ToolFactory = Callable[[SourceTracker], list[object]]
ToolLoop = Callable[..., Awaitable[object]]


class NjuQaAgent:
    """Two-stage evidence-first Agent."""

    def __init__(
        self,
        context: object,
        tools: ToolFactory,
        tool_loop: ToolLoop | None = None,
        docs_root: Path | None = None,
        index: Any = None,
        diagnostics: bool = False,
    ):
        self.context = context
        self.tools = tools
        self._tool_loop = tool_loop
        self.docs_root = docs_root
        self.index = index
        self.diagnostics = diagnostics

    async def answer(self, event: object, prompt: str) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        logger.info("NJU agent start: provider=%s prompt=%r", provider_id, prompt)
        if not provider_id:
            logger.warning("NJU agent: no chat provider configured")
            return NO_PROVIDER

        if _is_small_talk(prompt):
            return await self._answer_small_talk(event, prompt)

        tracker = SourceTracker()
        tracker.diagnostics = self.diagnostics
        await self._research_phase(event, prompt, tracker)

        if not tracker.evidence_excerpts:
            logger.info("NJU agent: no evidence excerpts after research")
            return NO_EVIDENCE

        self._log_evidence_summary(tracker)
        return await self._answer_phase(event, prompt, tracker)

    async def _answer_small_talk(self, event: object, prompt: str) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=SMALL_TALK_SYSTEM_PROMPT,
                tracker=SourceTracker(),
            )
        except Exception:
            logger.exception("NJU agent small-talk failed")
            return AGENT_ERROR
        text = str(getattr(response, "completion_text", "")).strip()
        logger.info("NJU agent small-talk: length=%d", len(text))
        return text

    async def _research_phase(
        self, event: object, prompt: str, tracker: SourceTracker
    ) -> None:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        if self.diagnostics:
            logger.info(
                "NJU agent research start: prompt=%r tools=%s",
                prompt,
                [getattr(t, "name", "") for t in self.tools(tracker)],
            )
        try:
            response = await self._run_tool_loop(
                event=event,
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
                tracker=tracker,
                max_steps=_MAX_RESEARCH_STEPS,
            )
        except Exception:
            logger.exception("NJU agent research phase failed")
            return
        # Ignore any natural-language answer produced during research; only the
        # evidence recorded by the tools matters.
        text = str(getattr(response, "completion_text", "")).strip()
        if self.diagnostics and text:
            logger.info("NJU agent research discarded text: %d chars", len(text))

    def _select_excerpts(
        self, tracker: SourceTracker, max_excerpts: int = _MAX_EVIDENCE
    ) -> list[EvidenceExcerpt]:
        """Choose the best concrete evidence for the answer prompt."""
        # Prefer direct reads over outlines and navigation summaries.
        type_order = {"read": 0, "details": 1, "outline": 2, "navigation": 3}
        scored = sorted(
            tracker.evidence_excerpts,
            key=lambda e: (
                type_order.get(e.evidence_type, 4),
                -e.score,
                -len(e.content),
            ),
        )
        return scored[:max_excerpts]

    async def _answer_phase(
        self, event: object, prompt: str, tracker: SourceTracker
    ) -> str:
        provider_id = await self.context.get_current_chat_provider_id(
            getattr(event, "unified_msg_origin")
        )
        self._log_evidence_summary(tracker)
        excerpts = self._select_excerpts(tracker)
        tracker.selected_excerpts = excerpts

        if not excerpts:
            logger.info("NJU agent: no selected excerpts for answer")
            return NO_EVIDENCE

        grounded_prompt = self._build_answer_prompt(prompt, excerpts)
        answer_tools = self._answer_tools(tracker)

        if self.diagnostics:
            logger.info(
                "NJU agent answer start: selected=%d ids=%s tools=%s",
                len(excerpts),
                [e.evidence_id for e in excerpts],
                [getattr(t, "name", "") for t in answer_tools],
            )

        for attempt in range(2):
            try:
                response = await self._run_tool_loop(
                    event=event,
                    chat_provider_id=provider_id,
                    prompt=grounded_prompt,
                    system_prompt=ANSWER_SYSTEM_PROMPT,
                    tracker=tracker,
                    tools=answer_tools,
                    max_steps=_MAX_ANSWER_STEPS,
                )
            except Exception:
                logger.exception("NJU agent answer phase failed")
                return AGENT_ERROR
            text = str(getattr(response, "completion_text", "")).strip()
            used_ids = _extract_used_evidence_ids(text)
            if used_ids or _is_pure_no_evidence(text):
                break
            logger.warning(
                "NJU agent: answer missing evidence markers (attempt %d)", attempt + 1
            )
            grounded_prompt = (
                grounded_prompt
                + "\n\n注意：你刚才的回答没有使用任何 [E#] 标记。"
                "请重新回答，并确保每个事实都附带 [E#] 标记。"
            )
        else:
            logger.error("NJU agent: answer still missing evidence markers")
            return SAFE_FAILURE

        return self._finalize_answer(text, excerpts, tracker.verified_urls)

    def _build_answer_prompt(
        self, question: str, excerpts: list[EvidenceExcerpt]
    ) -> str:
        parts = [f"请回答原问题：{question}\n"]
        parts.append(
            "你只能使用下面标记的证据回答问题。"
            "若某条证据明确标记 no_answer，只能说明该事项暂无可靠资料，"
            "不能用其他相邻材料推断；但不得因此忽略其他已有正面证据的问题部分。"
        )
        for excerpt in excerpts:
            loc = ""
            if excerpt.line_start is not None:
                end = excerpt.line_end if excerpt.line_end is not None else excerpt.line_start
                loc = f" 位置：第 {excerpt.line_start}—{end} 行"
            note = ""
            if excerpt.historical:
                note = "（历史资料）"
            if excerpt.qa_status == "no_answer":
                note += "（该证据明确说明暂无可靠资料）"
            header = (
                f"[{excerpt.evidence_id}]\n"
                f"来源：《{excerpt.title}》{loc}{note}\n"
                f"URL：{excerpt.url or 'n/a'}\n"
                f"内容：\n{excerpt.content}"
            )
            parts.append(header)
        return "\n\n".join(parts)

    def _answer_tools(self, tracker: SourceTracker) -> list[object]:
        """Return an empty tool set for the answer phase.

        The answer model must only use the evidence selected during research.
        Opening more tools here would allow new evidence that cannot be cited.
        """
        return []

    def _finalize_answer(
        self,
        text: str,
        excerpts: list[EvidenceExcerpt],
        verified_urls: set[str] | None = None,
    ) -> str:
        used_ids = _extract_used_evidence_ids(text)
        seen: set[str] = set()
        used: list[EvidenceExcerpt] = []
        for e in excerpts:
            if e.evidence_id in used_ids and e.evidence_id not in seen:
                seen.add(e.evidence_id)
                used.append(e)
        if not used and not _is_pure_no_evidence(text):
            logger.warning("NJU agent: no evidence markers in final answer")
            return SAFE_FAILURE

        # Strip internal markers from the visible answer.
        visible = re.sub(r"\[E\d+]", "", text).strip()
        if _is_pure_no_evidence(visible):
            logger.info("NJU agent: no-evidence answer, suppressing citations")
            return visible or NO_EVIDENCE

        # Remove any URLs the model hallucinated; keep URLs that appear in the
        # evidence content or in the cited excerpts.
        allowed_urls = set(verified_urls or set())
        allowed_urls.update(e.url for e in used if e.url)
        visible = _strip_unverified_urls(visible, allowed_urls)

        citations = self._build_citations(used)
        logger.info(
            "NJU agent: available=%d used=%d citations=%d",
            len(excerpts),
            len(used),
            len(citations),
        )
        if self.diagnostics:
            logger.info(
                "NJU agent final: available_ids=%s used_ids=%s citation_count=%d",
                [e.evidence_id for e in excerpts],
                used_ids,
                len(citations),
            )
        if not citations:
            return visible
        return f"{visible}\n\n参考来源：\n" + "\n".join(citations)

    def _build_citations(self, used: list[EvidenceExcerpt]) -> list[str]:
        seen: set[str] = set()
        citations: list[str] = []
        for excerpt in sorted(used, key=lambda e: e.evidence_id):
            key = excerpt.document_id or excerpt.url or excerpt.file_path
            if not key or key in seen:
                continue
            seen.add(key)
            citations.append(f"{len(citations) + 1}. 《{excerpt.title}》：{excerpt.url or excerpt.file_path}")
        return citations

    def _log_evidence_summary(self, tracker: SourceTracker) -> None:
        if not self.diagnostics:
            return
        logger.info("NJU evidence summary: candidates=%d excerpts=%d read=%d",
                    len(tracker.candidate_sources),
                    len(tracker.evidence_excerpts),
                    tracker.read_count)
        for i, cand in enumerate(tracker.candidate_sources[:20], 1):
            logger.info(
                "NJU candidate %d: title=%s path=%s score=%s",
                i,
                getattr(cand.document, "title", "?"),
                getattr(cand.document, "path", "?"),
                cand.score,
            )
        for e in tracker.evidence_excerpts:
            logger.info(
                "NJU evidence excerpt: id=%s type=%s doc=%s lines=%s:%s chars=%d qa=%s historical=%s",
                e.evidence_id,
                e.evidence_type,
                e.document_id,
                e.line_start,
                e.line_end,
                len(e.content),
                e.qa_status,
                e.historical,
            )

    async def _run_tool_loop(
        self, *, tracker: SourceTracker, **kwargs: object
    ) -> object:
        # Allow callers to override the tool set (e.g. answer phase).
        if "tools" not in kwargs:
            kwargs["tools"] = self.tools(tracker)
        max_steps = int(kwargs.pop("max_steps", 12))

        if self._tool_loop is not None:
            return await self._tool_loop(max_steps=max_steps, **kwargs)

        from astrbot.core.agent.tool import ToolSet

        # The real AstrBot tool loop does not accept our internal tracker; it is
        # passed to the tools factory above.
        kwargs.pop("tracker", None)
        return await self.context.tool_loop_agent(
            tools=ToolSet(kwargs.pop("tools")),
            max_steps=max_steps,
            tool_call_timeout=60,
            **kwargs,
        )

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
