"""AstrBot integration for NJU QA; domain logic lives in :mod:`nju_qa`."""

from __future__ import annotations

import asyncio
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
        self._sync_task: asyncio.Task | None = None

    async def initialize(self):
        self.index.open()
        self.chunk_store.open()

    async def _sync(self) -> str:
        result = await self.syncer.sync_all()
        return result.summary()

    @filter.command("nju")
    async def nju(self, event: AstrMessageEvent, question: str = ""):
        mark_command_handled(event)
        if question.strip().lower() == "help" or not question.strip():
            yield event.plain_result(
                "/nju <问题>：查询知识库\n/nju source <关键词>：查看来源\n本项目为非官方开源项目，与南京大学官方无隶属或授权关系。"
            )
            return
        if question.strip().lower().startswith("source "):
            yield event.plain_result(
                self._format_sources(await self.retriever.search(question.strip()[7:]))
            )
            return
        yield event.plain_result(await self.agent.answer(event, question))

    @filter.command("nju_sync")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_sync(self, event: AstrMessageEvent, action: str = ""):
        mark_command_handled(event)
        if action.lower() == "status":
            yield event.plain_result(self.syncer.status_text())
            return
        if self._sync_task and not self._sync_task.done():
            yield event.plain_result(
                "同步正在进行中；请使用 /nju_sync status 查看状态。"
            )
            return
        self._sync_task = asyncio.create_task(self._sync())
        yield event.plain_result("已启动后台同步；请使用 /nju_sync status 查看状态。")

    @filter.command("nju_index")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_index(self, event: AstrMessageEvent, action: str = ""):
        mark_command_handled(event)
        if action.lower() != "rebuild":
            yield event.plain_result("用法：/nju_index rebuild")
            return
        yield event.plain_result(await self.syncer.rebuild_index())

    @filter.command("nju_search")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_search(self, event: AstrMessageEvent, query: str = ""):
        mark_command_handled(event)
        yield event.plain_result(
            self.retriever.debug_text(await self.retriever.debug_search(query))
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        original = event.message_str or ""
        is_at_me = self._is_at_me(event)
        text = self._remove_at(event, original) if is_at_me else original
        routed = self.router.route(event, text, is_at_me)
        if not routed.should_handle or not routed.query:
            return
        mark_command_handled(event)
        try:
            yield event.plain_result(await self.agent.answer(event, routed.query))
        except Exception:
            logger.exception("NJU QA message handling failed")
            yield event.plain_result("处理问题时出错，请稍后重试。")

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
        await self.client.close()
        self.index.close()
        self.chunk_store.close()
        self.vector_index.close()
