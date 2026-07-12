"""AstrBot integration for NJU QA; domain logic lives in :mod:`nju_qa`."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

# AstrBot imports plugins as packages, so these must remain package-relative.
from .nju_qa.agent import NjuQaAgent
from .nju_qa.chunk_store import ChunkStore
from .nju_qa.config import PluginConfig
from .nju_qa.document_index import DocumentIndex
from .nju_qa.document_store import DocumentStore
from .nju_qa.formatting import markdown_to_plaintext
from .nju_qa.rate_limiter import RateLimiter, RateLimitState
from .nju_qa.retriever import HybridRetriever
from .nju_qa.routing import MessageRouter, mark_command_handled
from .nju_qa.sync_service import SyncService
from .nju_qa.tools import (
    GetDocDetailsTool,
    DocStatsTool,
    GrepLocalDocsTool,
    ListKnowledgeBasesTool,
    ListRepoDocsTool,
    ParseYuqueUrlTool,
    ReadDocTool,
    SearchDocsTool,
    SearchKnowledgeBaseTool,
)
from .nju_qa.yuque_client import YuqueClient
from .nju_qa.vector_index import ChunkVectorIndex


def _plain(event: AstrMessageEvent, text: str):
    """Send text stripped of Markdown markup for QQ plain-text chat."""
    return event.plain_result(markdown_to_plaintext(text))


@register("astrbot_plugin_nju_qa", "peace", "南京大学知识库问答助手", "0.1.0")
class NjuQaPlugin(Star):
    """Explicitly triggered NJU knowledge-base Q&A plugin."""

    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context, self.config = context, PluginConfig.from_mapping(config or {})
        data_dir = Path(
            getattr(self, "data_dir", Path("data") / "astrbot_plugin_nju_qa")
        )
        self.store = DocumentStore(data_dir / "documents")
        self.index = DocumentIndex(data_dir / "nju_qa.sqlite3")
        self.chunk_store = ChunkStore(data_dir / "chunks.sqlite3")
        self.client = YuqueClient(self.config.yuque_token, self.config.yuque_base_url)
        self.vector_index = ChunkVectorIndex(
            data_dir / "vectors", self.config.embedding_model
        )
        self.syncer = SyncService(
            self.config,
            self.client,
            self.store,
            self.index,
            chunk_store=self.chunk_store,
            vector_index=self.vector_index,
        )
        self.retriever = HybridRetriever(
            self.index,
            self.config,
            chunk_store=self.chunk_store,
            vector_index=self.vector_index,
        )
        self.agent = NjuQaAgent(
            self.context,
            lambda tracker: [
                SearchKnowledgeBaseTool(retriever=self.retriever, tracker=tracker),
                GrepLocalDocsTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                ReadDocTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                SearchDocsTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                GetDocDetailsTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                ParseYuqueUrlTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                ListKnowledgeBasesTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                ListRepoDocsTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
                DocStatsTool(
                    index=self.index, docs_root=self.store.root, tracker=tracker
                ),
            ],
            docs_root=self.store.root,
        )
        self.router = MessageRouter(
            self.config.wake_words,
            self.config.enable_private_chat,
            self.config.enable_group_at,
        )
        self.rate_limiter = RateLimiter(
            group_max=self.config.group_rate_limit,
            group_window_seconds=self.config.group_rate_limit_window,
            private_max=self.config.private_rate_limit,
            private_window_seconds=self.config.private_rate_limit_window,
        )
        self._sync_task: asyncio.Task | None = None
        self._rebuild_task: asyncio.Task | None = None

    async def initialize(self):
        self.index.open()
        self.chunk_store.open()

    async def _sync(self) -> str:
        result = await self.syncer.sync_all()
        return result.summary()

    async def _rebuild(self) -> str:
        return await self.syncer.rebuild_index()

    @filter.command("nju")
    async def nju(self, event: AstrMessageEvent, question: str = ""):
        mark_command_handled(event)
        rl_state = self._check_rate_limit(event)
        if rl_state:
            if not rl_state.silent:
                yield _plain(event, self._rate_limit_message(rl_state))
            return
        # AstrBot strips the leading '/' from event.message_str for commands.
        text = (getattr(event, "message_str", None) or "").strip() or (
            "nju " + question
        ).strip()

        source_match = re.match(r"^nju\s+source\s+(.+)$", text, re.IGNORECASE)
        if source_match:
            keyword = source_match.group(1).strip()
            if not keyword:
                yield _plain(event,"用法：/nju source <关键词>")
                return
            yield _plain(event,
                self._format_sources(await self.retriever.search(keyword))
            )
            return

        if re.match(r"^nju(\s+help)?$", text, re.IGNORECASE):
            yield _plain(event,
                "/nju <问题>：查询知识库\n"
                "/nju source <关键词>：查看来源\n"
                "/nju_grep <关键词>：全文搜索本地文档\n"
                "/nju_sync / /nju_index / /nju_search：管理员命令\n"
                "本项目为非官方开源项目，与南京大学官方无隶属或授权关系。"
            )
            return

        # Regular question: strip the nju prefix.
        query = re.sub(r"^nju\s*", "", text, flags=re.IGNORECASE).strip()
        logger.info("NJU command parsed query: %r", query)
        try:
            answer = await self.agent.answer(event, query)
        except Exception as exc:
            logger.exception("NJU QA agent failed")
            yield _plain(event,f"检索失败：{exc}")
            return
        logger.info("NJU command answer length=%d first_100=%r", len(answer), answer[:100])
        yield _plain(event,answer)

    @filter.command("nju_debug")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_debug(self, event: AstrMessageEvent, question: str = ""):
        """Diagnostic command to inspect how AstrBot parses /nju messages."""
        mark_command_handled(event)
        text = (getattr(event, "message_str", None) or "").strip()
        source_re = r"^nju\s+source\s+"
        matched = bool(re.match(source_re, text, re.IGNORECASE))
        yield _plain(event,
            f"message_str: {repr(text)}\n"
            f"handler_arg: {repr(question)}\n"
            f"matched_source: {matched}"
        )

    @filter.command("nju_grep")
    async def nju_grep(self, event: AstrMessageEvent, keywords: str = ""):
        mark_command_handled(event)
        rl_state = self._check_rate_limit(event)
        if rl_state:
            if not rl_state.silent:
                yield _plain(event, self._rate_limit_message(rl_state))
            return
        if not keywords.strip():
            yield _plain(event,"用法：/nju_grep <空格分隔的关键词>")
            return
        tool = GrepLocalDocsTool(index=self.index, docs_root=self.store.root)
        result = await tool._run(keywords)
        # If the exact phrase yields nothing, try splitting long Chinese queries
        # into overlapping 2-character terms (e.g. "确认录取" → "确认 录取").
        if not result.get("results"):
            cleaned = keywords.strip().replace(" ", "")
            if len(cleaned) >= 4:
                split_terms = " ".join(
                    cleaned[i : i + 2] for i in range(0, len(cleaned) - 1, 2)
                )
                result = await tool._run(split_terms)
        if not result.get("results"):
            yield _plain(event,"本地文档中未找到匹配内容。")
            return
        lines = [f"共找到 {result['count']} 条："]
        for i, hit in enumerate(result["results"][:10], 1):
            snippet = hit.get("snippet", "").replace("\n", " ")[:200]
            source_url = hit.get("source_url") or hit.get("url") or ""
            lines.append(
                f"{i}. 《{hit.get('title')}》：{source_url}\n   {snippet}..."
            )
        yield _plain(event,"\n".join(lines))

    @filter.command("nju_sync")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_sync(self, event: AstrMessageEvent, action: str = ""):
        mark_command_handled(event)
        if action.lower() == "status":
            yield _plain(event,self.syncer.status_text())
            return
        if self._sync_task and not self._sync_task.done():
            yield _plain(event,
                "同步正在进行中；请使用 /nju_sync status 查看状态。"
            )
            return
        self._sync_task = asyncio.create_task(self._sync())
        yield _plain(event,"已启动后台同步；请使用 /nju_sync status 查看状态。")

    @filter.command("nju_index")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_index(self, event: AstrMessageEvent, action: str = ""):
        mark_command_handled(event)
        if action.lower() != "rebuild":
            yield _plain(event,"用法：/nju_index rebuild")
            return
        if self._rebuild_task is not None and not self._rebuild_task.done():
            yield _plain(event,
                "索引重建正在进行中；请使用 /nju_sync status 查看状态。"
            )
            return
        self._rebuild_task = asyncio.create_task(self._rebuild())
        yield _plain(event,
            "已启动后台索引重建；请使用 /nju_sync status 查看进度和结果。"
        )

    @filter.command("nju_search")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_search(self, event: AstrMessageEvent, query: str = ""):
        mark_command_handled(event)
        if not query.strip():
            yield _plain(event,"用法：/nju_search <查询词>")
            return
        yield _plain(event,
            self.retriever.debug_text(await self.retriever.debug_search(query))
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        original = self._raw_message_text(event)
        # AstrBot strips the leading '/' from event.message_str before command
        # handlers run, so ALL-message listeners see "audit ok" instead of
        # "/audit ok". Ignore raw command-like messages to avoid conflicting
        # with other plugins (e.g. astrbot_plugin_nju_qq_audit).
        if original.lstrip().startswith("/"):
            return
        is_at_me = self._is_at_me(event)
        text = self._remove_at(event, original) if is_at_me else original
        routed = self.router.route(event, text, is_at_me)
        if not routed.should_handle or not routed.query:
            return
        mark_command_handled(event)
        rl_state = self._check_rate_limit(event)
        if rl_state:
            if not rl_state.silent:
                yield _plain(event, self._rate_limit_message(rl_state))
            return
        try:
            yield _plain(event,await self.agent.answer(event, routed.query))
        except Exception:
            logger.exception("NJU QA message handling failed")
            yield _plain(event,"处理问题时出错，请稍后重试。")

    @staticmethod
    def _raw_message_text(event: AstrMessageEvent) -> str:
        """Return the original plain text before AstrBot strips the command '/'."""
        try:
            from astrbot.api import message_components as comp

            return "".join(
                part.text
                for part in event.message_obj.message
                if isinstance(part, comp.Plain)
            ).strip()
        except (AttributeError, ImportError):
            return (event.message_str or "").strip()

    @staticmethod
    def _format_sources(results) -> str:
        if not results:
            return "知识库中暂未找到可靠答案"
        lines = ["参考来源："]
        lines.extend(
            f"{number}. 《{result.document.title}》：{result.document.url}"
            for number, result in enumerate(results, 1)
        )
        return "\n".join(lines)

    @staticmethod
    def _chat_key(event: AstrMessageEvent) -> tuple[bool, str]:
        """Return (is_group, key) for rate-limit tracking."""
        group_id = ""
        getter = getattr(event, "get_group_id", None)
        if callable(getter):
            group_id = getter() or ""
        is_group = group_id not in (None, "")
        if is_group:
            return True, str(group_id)

        sender_id = ""
        sender_getter = getattr(event, "get_sender_id", None)
        if callable(sender_getter):
            sender_id = sender_getter() or ""
        if sender_id:
            return False, f"private:{sender_id}"

        origin = getattr(event, "unified_msg_origin", "") or ""
        if origin:
            return False, f"private:{origin}"

        return False, f"private:{id(event)}"

    def _check_rate_limit(self, event: AstrMessageEvent) -> RateLimitState | None:
        """Return RateLimitState only when the event is blocked."""
        is_group, key = self._chat_key(event)
        allowed, state = self.rate_limiter.is_allowed(key, is_group)
        return None if allowed else state

    @staticmethod
    def _rate_limit_message(state: RateLimitState) -> str:
        window_min = state.window_seconds // 60
        if state.is_group:
            return (
                f"本群已达到当前时段的提问上限（{state.max_count} 次/{window_min} 分钟），"
                "请稍后再试，或私聊我提问。"
            )
        return (
            f"你已达到当前时段的提问上限（{state.max_count} 次/{window_min} 分钟），请稍后再试。"
        )

    def _is_at_me(self, event: AstrMessageEvent) -> bool:
        try:
            from astrbot.api import message_components as comp

            return any(
                isinstance(x, comp.At) and str(x.qq) == str(event.get_self_id())
                for x in event.message_obj.message
            )
        except (AttributeError, ImportError):
            return False

    @staticmethod
    def _remove_at(event: AstrMessageEvent, fallback: str) -> str:
        try:
            from astrbot.api import message_components as comp

            text = "".join(
                part.text
                for part in event.message_obj.message
                if isinstance(part, comp.Plain)
            ).strip()
            return text or fallback
        except (AttributeError, ImportError):
            return fallback

    async def terminate(self):
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            await asyncio.gather(self._sync_task, return_exceptions=True)
        if self._rebuild_task and not self._rebuild_task.done():
            self._rebuild_task.cancel()
            await asyncio.gather(self._rebuild_task, return_exceptions=True)
        await self.client.close()
        self.index.close()
        self.chunk_store.close()
        self.vector_index.close()
