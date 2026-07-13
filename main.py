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
from .nju_qa.table_renderer import clean_table_images, ensure_cjk_font, render_tables_as_images
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


def _build_rich_result(event: AstrMessageEvent, segments: list[tuple[str, str]]):
    """Build a message chain from text/image segments."""
    from astrbot.api import message_components as comp

    chain: list[object] = []
    for seg_type, content in segments:
        if seg_type == "text":
            if content.strip():
                chain.append(comp.Plain(content))
        elif seg_type == "image":
            chain.append(comp.Image.fromFileSystem(content))
    if not chain:
        return event.plain_result("")
    if len(chain) == 1 and isinstance(chain[0], comp.Plain):
        return event.plain_result(chain[0].text)
    return event.chain_result(chain)


@register("astrbot_plugin_nju_qa", "peace", "南京大学知识库问答助手", "0.2.0")
class NjuQaPlugin(Star):
    """Explicitly triggered NJU knowledge-base Q&A plugin."""

    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context, self.config = context, PluginConfig.from_mapping(config or {})
        data_dir = Path(
            getattr(self, "data_dir", Path("data") / "astrbot_plugin_nju_qa")
        )
        self._data_dir = data_dir
        self._table_image_dir = data_dir / "table_images"
        self._resolved_font_path: str | None = None
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
        clean_table_images(self._table_image_dir)
        self._resolved_font_path = await ensure_cjk_font(
            self._data_dir,
            self.config.table_font_path or None,
            self.config.auto_download_table_font,
            download_timeout=self.config.table_font_download_timeout,
        )

    async def _sync(self) -> str:
        result = await self.syncer.sync_all()
        return result.summary()

    async def _rebuild(self) -> str:
        return await self.syncer.rebuild_index()

    @filter.command("nju")
    async def nju(self, event: AstrMessageEvent, question: str = ""):
        # AstrBot strips the leading '/' from event.message_str for commands.
        text = (getattr(event, "message_str", None) or "").strip() or (
            "nju " + question
        ).strip()
        # Ignore command messages that are not actually /nju (some AstrBot builds
        # may route all slash commands to every command handler).
        if not re.match(r"^nju(\s+.*)?$", text, re.IGNORECASE):
            return
        mark_command_handled(event)

        source_match = re.match(r"^nju\s+source\s+(.+)$", text, re.IGNORECASE)
        if source_match:
            keyword = source_match.group(1).strip()
            if not keyword:
                yield self._rich_result(event, "用法：/nju source <关键词>")
                return
            yield self._rich_result(
                event, self._format_sources(await self.retriever.search(keyword))
            )
            return

        if re.match(r"^nju(\s+help)?$", text, re.IGNORECASE):
            yield self._rich_result(
                event,
                "/nju <问题>：查询知识库\n"
                "/nju source <关键词>：查看来源\n"
                "/nju_grep <关键词>：全文搜索本地文档\n"
                "/nju_sync / /nju_index / /nju_search：管理员命令\n"
                "本项目为非官方开源项目，与南京大学官方无隶属或授权关系。",
            )
            return

        # Regular question: rate limiting and answering live in _answer_question
        # so /nju, @-mentions, wake words and private chat all share one path.
        query = re.sub(r"^nju\s*", "", text, flags=re.IGNORECASE).strip()
        logger.info("NJU command parsed query: %r", query)
        async for result in self._answer_question(event, query):
            yield result

    @filter.command("nju_debug")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_debug(self, event: AstrMessageEvent, question: str = ""):
        """Diagnostic command to inspect how AstrBot parses /nju messages."""
        text = (getattr(event, "message_str", None) or "").strip()
        if not text.lower().startswith("nju_debug"):
            return
        mark_command_handled(event)
        source_re = r"^nju\s+source\s+"
        matched = bool(re.match(source_re, text, re.IGNORECASE))
        yield self._rich_result(event,
            f"message_str: {repr(text)}\n"
            f"handler_arg: {repr(question)}\n"
            f"matched_source: {matched}"
        )

    @filter.command("nju_grep")
    async def nju_grep(self, event: AstrMessageEvent, keywords: str = ""):
        text = (getattr(event, "message_str", None) or "").strip()
        if not text.lower().startswith("nju_grep"):
            return
        mark_command_handled(event)
        rl_state = self._check_rate_limit(event)
        if rl_state:
            if not rl_state.silent:
                yield self._rich_result(event, self._rate_limit_message(rl_state))
            return
        if not keywords.strip():
            yield self._rich_result(event,"用法：/nju_grep <空格分隔的关键词>")
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
            yield self._rich_result(event,"本地文档中未找到匹配内容。")
            return
        lines = [f"共找到 {result['count']} 条："]
        for i, hit in enumerate(result["results"][:10], 1):
            snippet = hit.get("snippet", "").replace("\n", " ")[:200]
            source_url = hit.get("source_url") or hit.get("url") or ""
            lines.append(
                f"{i}. 《{hit.get('title')}》：{source_url}\n   {snippet}..."
            )
        yield self._rich_result(event,"\n".join(lines))

    @filter.command("nju_sync")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_sync(self, event: AstrMessageEvent, action: str = ""):
        text = (getattr(event, "message_str", None) or "").strip()
        if not text.lower().startswith("nju_sync"):
            return
        mark_command_handled(event)
        if action.lower() == "status":
            yield self._rich_result(event,self.syncer.status_text())
            return
        if self._sync_task and not self._sync_task.done():
            yield self._rich_result(event,
                "同步正在进行中；请使用 /nju_sync status 查看状态。"
            )
            return
        self._sync_task = asyncio.create_task(self._sync())
        yield self._rich_result(event,"已启动后台同步；请使用 /nju_sync status 查看状态。")

    @filter.command("nju_index")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_index(self, event: AstrMessageEvent, action: str = ""):
        text = (getattr(event, "message_str", None) or "").strip()
        if not text.lower().startswith("nju_index"):
            return
        mark_command_handled(event)
        if action.lower() != "rebuild":
            yield self._rich_result(event,"用法：/nju_index rebuild")
            return
        if self._rebuild_task is not None and not self._rebuild_task.done():
            yield self._rich_result(event,
                "索引重建正在进行中；请使用 /nju_sync status 查看状态。"
            )
            return
        self._rebuild_task = asyncio.create_task(self._rebuild())
        yield self._rich_result(event,
            "已启动后台索引重建；请使用 /nju_sync status 查看进度和结果。"
        )

    @filter.command("nju_search")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_search(self, event: AstrMessageEvent, query: str = ""):
        text = (getattr(event, "message_str", None) or "").strip()
        if not text.lower().startswith("nju_search"):
            return
        mark_command_handled(event)
        if not query.strip():
            yield self._rich_result(event,"用法：/nju_search <查询词>")
            return
        yield self._rich_result(event,
            self.retriever.debug_text(await self.retriever.debug_search(query))
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        # If a command handler (ours or another plugin) has already stopped the
        # event, do nothing further.
        if self._is_event_stopped(event):
            return
        # If AstrBot matched a registered command (ours or another plugin), let
        # that handler run.  We must not reply or stop the event here.
        if self._has_matched_registered_command(event):
            return
        original = self._raw_message_text(event)
        # Unknown slash messages should not wake the default LLM, but they also
        # must not be stopped so that other plugins' ALL-message handlers can
        # still process them.
        if self._is_unknown_slash_message(event, original):
            self._suppress_default_llm(event)
            return
        is_at_me = self._is_at_me(event)
        text = self._remove_at(event, original) if is_at_me else original
        routed = self.router.route(event, text, is_at_me)
        if not routed.should_handle or not routed.query:
            return
        mark_command_handled(event)
        async for result in self._answer_question(event, routed.query):
            yield result

    @staticmethod
    def _raw_message_text(event: AstrMessageEvent) -> str:
        """Return the original plain text before AstrBot strips the command prefix."""
        try:
            from astrbot.api import message_components as comp

            return "".join(
                part.text
                for part in NjuQaPlugin._get_message_chain(event)
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

    def _rich_result(self, event: AstrMessageEvent, text: str):
        """Return plain text or a text+image chain if the answer contains tables."""
        if not self.config.render_tables_as_images:
            return _plain(event, text)
        font_path = self.config.table_font_path or self._resolved_font_path
        segments = render_tables_as_images(text, self._table_image_dir, font_path=font_path)
        return _build_rich_result(event, segments)

    @staticmethod
    def _is_event_stopped(event: AstrMessageEvent) -> bool:
        """Return True if a previous handler has already stopped the event."""
        getter = getattr(event, "is_stopped", None)
        if callable(getter):
            return getter()
        return bool(getattr(event, "stopped", False))

    @staticmethod
    def _has_matched_registered_command(event: AstrMessageEvent) -> bool:
        """Return True if AstrBot has matched a registered command handler.

        This relies on metadata written by WakingCheckStage.  It does not rely
        on ``event.is_command`` or ``event.is_stopped()``, because those flags
        may be set for any ``/`` wake-prefix message or may be missing.
        """
        # AstrBot may set this flag directly for wake-prefix commands.
        if event.get_extra("astrbot_known_wake_prefix_command") is True:
            return True

        parsed = event.get_extra("handlers_parsed_params")
        if isinstance(parsed, dict) and parsed:
            # Any non-empty parsed-params dict means at least one command-filter
            # handler matched (including NJU QA's own /nju).  Let those handlers
            # run; on_message should not interfere.
            return True

        activated = event.get_extra("activated_handlers")
        if isinstance(activated, (list, tuple)):
            for handler in activated:
                if not handler:
                    continue
                for filter_ in getattr(handler, "event_filters", []) or []:
                    if type(filter_).__name__ in ("CommandFilter", "CommandGroupFilter"):
                        return True

        return False

    @staticmethod
    def _get_message_chain(event: AstrMessageEvent) -> list:
        """Return the message chain, or an empty list on error."""
        try:
            return event.message_obj.message
        except Exception:
            return []

    def _wake_prefixes(self) -> list[str]:
        """Return the configured wake prefixes, defaulting to ["/"]."""
        ctx = getattr(self, "context", None)
        cfg = getattr(ctx, "astrbot_config", None) or getattr(ctx, "config", None) or {}
        prefixes = cfg.get("wake_prefix", ["/"])
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        return list(prefixes) or ["/"]

    def _is_unknown_slash_message(
        self, event: AstrMessageEvent, original: str
    ) -> bool:
        """Return True for an unmatched wake-prefix slash message.

        We must block the default LLM for these messages without stopping the
        event, so other plugins' ALL-message handlers can still process them.
        """
        # Fast path: the original message chain still starts with a wake prefix.
        prefixes = self._wake_prefixes()
        if any(original.startswith(p) for p in prefixes):
            return True

        # Fallback: some adapters strip the prefix from message_obj.  If the
        # event was woken by a wake prefix in a group (not @, not private) and
        # no registered command matched, treat it as an unknown slash message.
        if not getattr(event, "is_at_or_wake_command", False):
            return False
        if event.is_private_chat():
            return False
        messages = self._get_message_chain(event)
        if messages:
            from astrbot.api import message_components as comp

            at_types = tuple(
                cls
                for name in ("At", "AtAll", "Reply")
                if (cls := getattr(comp, name, None)) is not None
            )
            first = messages[0]
            if at_types and isinstance(first, at_types):
                return False
        return True

    @staticmethod
    def _suppress_default_llm(event: AstrMessageEvent) -> None:
        """Tell AstrBot not to fall back to the default LLM for this event.

        AstrBot's ProcessStage calls the default LLM when
        ``not event.call_llm`` is true for an @/wake message.  Therefore
        ``should_call_llm(True)`` (setting ``call_llm=True``) is the value
        that actually blocks the fallback.  The naming is inverted relative
        to the public method name.
        """
        setter = getattr(event, "should_call_llm", None)
        if callable(setter):
            setter(True)
        else:
            # Fallback for older AstrBot builds that expose the attribute
            # directly: set it so the ProcessStage guard sees a truthy value.
            try:
                event.call_llm = True
            except Exception:
                pass

    async def _answer_question(self, event: AstrMessageEvent, query: str):
        """Shared Q&A entry used by /nju, @-mentions, wake words and private chat."""
        rl_state = self._check_rate_limit(event)
        if rl_state:
            if not rl_state.silent:
                yield self._rich_result(event, self._rate_limit_message(rl_state))
            return
        try:
            answer = await self.agent.answer(event, query)
        except Exception as exc:
            logger.exception("NJU QA agent failed")
            yield self._rich_result(event, f"检索失败：{exc}")
            return
        yield self._rich_result(event, answer)

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
