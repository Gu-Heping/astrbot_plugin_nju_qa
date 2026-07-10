"""AstrBot integration for NJU QA; domain logic lives in :mod:`nju_qa`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from nju_qa.answer_service import AnswerService
from nju_qa.config import PluginConfig
from nju_qa.document_index import DocumentIndex
from nju_qa.document_store import DocumentStore
from nju_qa.retriever import HybridRetriever
from nju_qa.sync_service import SyncService
from nju_qa.yuque_client import YuqueClient


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
        self.client = YuqueClient(self.config.yuque_token, self.config.yuque_base_url)
        self.syncer = SyncService(self.config, self.client, self.store, self.index)
        self.answers = AnswerService(
            HybridRetriever(self.index, self.config), self._call_llm
        )
        self._sync_task: asyncio.Task | None = None

    async def initialize(self):
        self.index.open()

    async def _call_llm(self, prompt: str, system_prompt: str) -> str:
        provider = self.context.get_using_provider()
        response = await provider.text_chat(
            prompt=prompt, context=[], system_prompt=system_prompt
        )
        return response.completion_text.strip()

    async def _sync(self) -> str:
        result = await self.syncer.sync_all()
        return result.summary()

    @filter.command("nju")
    async def nju(self, event: AstrMessageEvent, question: str = ""):
        if question.strip().lower() == "help" or not question.strip():
            yield event.plain_result(
                "/nju <问题>：查询知识库\n/nju source <关键词>：查看来源\n本项目为非官方开源项目，与南京大学官方无隶属或授权关系。"
            )
            return
        if question.strip().lower().startswith("source "):
            yield event.plain_result(
                self.answers.format_source_results(
                    await self.answers.source_results(question.strip()[7:])
                )
            )
            return
        yield event.plain_result(await self.answers.answer(question))

    @filter.command("nju_sync")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_sync(self, event: AstrMessageEvent, action: str = ""):
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
        if action.lower() != "rebuild":
            yield event.plain_result("用法：/nju_index rebuild")
            return
        yield event.plain_result(await self.syncer.rebuild_index())

    @filter.command("nju_search")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def nju_search(self, event: AstrMessageEvent, query: str = ""):
        yield event.plain_result(
            self.answers.format_source_results(await self.answers.source_results(query))
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        if not text or text.startswith("/"):
            return
        group = event.get_group_id() is not None
        if group and not self.config.enable_group_at:
            return
        if not group and not self.config.enable_private_chat:
            return
        if group and not self._is_at_me(event):
            lowered = text.casefold()
            wake = next(
                (
                    word
                    for word in self.config.wake_words
                    if lowered.startswith(word.casefold())
                ),
                None,
            )
            if not wake:
                return
            text = text[len(wake) :].lstrip(" ，,：:")
        if not text:
            return
        try:
            yield event.plain_result(await self.answers.answer(text))
            event.stop_event()
        except Exception:
            logger.exception("NJU QA message handling failed")
            yield event.plain_result("处理问题时出错，请稍后重试。")

    def _is_at_me(self, event: AstrMessageEvent) -> bool:
        try:
            from astrbot.api import message_components as comp

            return any(
                isinstance(x, comp.At) and str(x.qq) == str(event.get_self_id())
                for x in event.message_obj.message
            )
        except (AttributeError, ImportError):
            return False

    async def terminate(self):
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            await asyncio.gather(self._sync_task, return_exceptions=True)
        await self.client.close()
        self.index.close()
