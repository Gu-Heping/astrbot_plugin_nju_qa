"""Deterministic tests for the group whitelist access-control feature."""

from __future__ import annotations

import asyncio
import types

import pytest

from nju_qa.tools import GrepLocalDocsTool
from tests.helpers import _FakeEvent, _collect, _make_plugin


def _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=(), **kwargs):
    """Create a plugin with the group whitelist configured."""
    return _make_plugin(
        plugin_class,
        tmp_path,
        enable_group_whitelist=enabled,
        group_whitelist=whitelist,
        **kwargs,
    )


def _group_event(text, group_id="g1", raw_text=None, message_chain=None, **kwargs):
    return _FakeEvent(
        group_id=group_id,
        text=text,
        raw_text=raw_text if raw_text is not None else text,
        message_chain=message_chain,
        **kwargs,
    )


def _at_chain(text: str):
    from astrbot.api import message_components as comp

    return [comp.At("bot"), comp.Plain(text)]


def _private_event(text, message_chain=None):
    return _FakeEvent(
        group_id="",
        sender_id="u1",
        text=text,
        raw_text=text,
        is_at_or_wake_command=True,
        message_chain=message_chain,
    )


def _raise_if_called(name: str):
    def _sync(*args, **kwargs):
        raise AssertionError(f"{name} must not be called for unauthorized group")

    async def _async(*args, **kwargs):
        raise AssertionError(f"{name} must not be called for unauthorized group")

    return _async if name.startswith("async_") else _sync


# ---------------------------------------------------------------------------
# Normal message routing
# ---------------------------------------------------------------------------


def test_whitelist_disabled_allows_group_wake_word(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=False)
    event = _group_event("南大助手 hello", group_id="g1")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_whitelist_disabled_allows_group_at(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=False)
    event = _group_event(
        "hello", group_id="g1", message_chain=_at_chain("hello")
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_whitelist_allows_whitelisted_group_wake_word(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event("南大助手 hello", group_id="g1")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_whitelist_allows_whitelisted_group_at(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event(
        "hello", group_id="g1", message_chain=_at_chain("hello")
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_whitelist_denies_non_whitelisted_group_wake_word(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event("南大助手 hello", group_id="g2")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


def test_whitelist_denies_non_whitelisted_group_at(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event(
        "hello", group_id="g2", message_chain=_at_chain("hello")
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


def test_whitelist_empty_list_denies_all_groups(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=[])
    event = _group_event("南大助手 hello", group_id="g1")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []


def test_whitelist_does_not_affect_plain_group_message(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event("hello", group_id="g2")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert plugin.agent.calls == []


def test_whitelist_allows_unknown_slash_to_pass_through(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event(
        "你好",
        group_id="g2",
        raw_text="/你好",
        is_command=False,
        is_at_or_wake_command=True,
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == [True]
    assert plugin.agent.calls == []


# ---------------------------------------------------------------------------
# Private chat
# ---------------------------------------------------------------------------


def test_whitelist_allows_private_message(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=[])
    event = _private_event("hello")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_whitelist_allows_private_at(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=[])
    event = _private_event(
        "hello", message_chain=_at_chain("hello")
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_whitelist_allows_private_nju_command(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=[])
    event = _private_event("nju hello")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


# ---------------------------------------------------------------------------
# Command handling for unauthorized groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,event_text,handler_kwargs,side_effect_target",
    [
        ("nju", "nju hello", {}, None),
        ("nju", "nju help", {}, None),
        ("nju", "nju source 宿舍", {}, "retriever.search"),
        ("nju_grep", "nju_grep 关键词", {"keywords": "关键词"}, "grep"),
        ("nju_sync", "nju_sync status", {"action": "status"}, "syncer.sync_all"),
        ("nju_index", "nju_index rebuild", {"action": "rebuild"}, "syncer.rebuild_index"),
        ("nju_search", "nju_search 查询词", {"query": "查询词"}, "retriever.debug_search"),
        ("nju_debug", "nju_debug test", {}, None),
    ],
)
def test_unauthorized_group_commands_are_silent(
    command, event_text, handler_kwargs, side_effect_target, plugin_class, tmp_path, monkeypatch
):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event(event_text, group_id="g2")

    if side_effect_target == "retriever.search":
        monkeypatch.setattr(
            plugin.retriever, "search", _raise_if_called("async_retriever.search")
        )
    elif side_effect_target == "retriever.debug_search":
        monkeypatch.setattr(
            plugin.retriever, "debug_search", _raise_if_called("async_retriever.debug_search")
        )
    elif side_effect_target == "syncer.sync_all":
        monkeypatch.setattr(
            plugin.syncer, "sync_all", _raise_if_called("async_syncer.sync_all")
        )
    elif side_effect_target == "syncer.rebuild_index":
        monkeypatch.setattr(
            plugin.syncer, "rebuild_index", _raise_if_called("async_syncer.rebuild_index")
        )
    elif side_effect_target == "grep":
        monkeypatch.setattr(
            GrepLocalDocsTool, "_run", _raise_if_called("async_GrepLocalDocsTool._run")
        )

    handler = getattr(plugin, command)
    results = asyncio.run(_collect(handler(event, **handler_kwargs)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


# ---------------------------------------------------------------------------
# Command handling for authorized groups
# ---------------------------------------------------------------------------


def test_authorized_group_nju_answers(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event("nju hello", group_id="g1")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_authorized_group_nju_help_replies(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event("nju help", group_id="g1")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "/nju <问题>" in results[0]
    assert plugin.agent.calls == []


def test_authorized_group_nju_source_searches(plugin_class, tmp_path, monkeypatch):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    calls = []

    async def fake_search(query, **scope):
        calls.append(query)
        return []

    monkeypatch.setattr(plugin.retriever, "search", fake_search)
    event = _group_event("nju source 宿舍", group_id="g1")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert calls == ["宿舍"]
    assert "知识库中暂未找到可靠答案" in results[0]


def test_authorized_group_nju_grep_runs(plugin_class, tmp_path, monkeypatch):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])

    async def fake_run(self, keywords):
        return {
            "count": 1,
            "results": [
                {
                    "title": "宿舍",
                    "snippet": "四人间",
                    "url": "https://yuque.test/dorm",
                }
            ],
        }

    monkeypatch.setattr(GrepLocalDocsTool, "_run", fake_run)
    event = _group_event("nju_grep 宿舍", group_id="g1")
    results = asyncio.run(_collect(plugin.nju_grep(event, keywords="宿舍")))

    assert event.stopped is True
    assert len(results) == 1
    assert "四人间" in results[0]


def test_authorized_group_nju_sync_starts_task(plugin_class, tmp_path, monkeypatch):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])

    async def fake_sync_all():
        return types.SimpleNamespace(summary=lambda: "ok")

    monkeypatch.setattr(plugin.syncer, "sync_all", fake_sync_all)
    event = _group_event("nju_sync", group_id="g1")
    results = asyncio.run(_collect(plugin.nju_sync(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "已启动后台同步" in results[0]
    if plugin._sync_task and not plugin._sync_task.done():
        plugin._sync_task.cancel()


def test_authorized_group_nju_index_starts_task(plugin_class, tmp_path, monkeypatch):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])

    async def fake_rebuild():
        return "done"

    monkeypatch.setattr(plugin.syncer, "rebuild_index", fake_rebuild)
    event = _group_event("nju_index rebuild", group_id="g1")
    results = asyncio.run(_collect(plugin.nju_index(event, action="rebuild")))

    assert event.stopped is True
    assert len(results) == 1
    assert "已启动后台索引重建" in results[0]
    if plugin._rebuild_task and not plugin._rebuild_task.done():
        plugin._rebuild_task.cancel()


def test_authorized_group_nju_search_searches(plugin_class, tmp_path, monkeypatch):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    calls = []

    async def fake_debug_search(query, **scope):
        calls.append(query)
        return {
            "mode": "hybrid",
            "embedding_available": False,
            "query_terms": [query],
            "chunk_count": 0,
            "vector_count": 0,
            "threshold": 0.25,
            "keyword_candidates": [],
            "vector_candidates": [],
            "merged": [],
            "selected": [],
        }

    monkeypatch.setattr(plugin.retriever, "debug_search", fake_debug_search)
    event = _group_event("nju_search 查询词", group_id="g1")
    results = asyncio.run(_collect(plugin.nju_search(event, query="查询词")))

    assert event.stopped is True
    assert len(results) == 1
    assert calls == ["查询词"]
    assert "检索模式" in results[0]


def test_authorized_group_nju_debug_replies(plugin_class, tmp_path):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    event = _group_event("nju_debug test", group_id="g1")
    results = asyncio.run(_collect(plugin.nju_debug(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "njudebug test" in results[0]


# ---------------------------------------------------------------------------
# Side-effect assertions for the unauthorized /nju message path
# ---------------------------------------------------------------------------


def test_unauthorized_group_message_has_no_side_effects(
    plugin_class, tmp_path, monkeypatch
):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    monkeypatch.setattr(
        plugin.agent, "answer", _raise_if_called("async_agent.answer")
    )
    monkeypatch.setattr(
        plugin.retriever, "search", _raise_if_called("async_retriever.search")
    )

    event = _group_event("南大助手 hello", group_id="g2")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets
    assert event.should_call_llm_calls == []


# ---------------------------------------------------------------------------
# Private-chat disable semantics
# ---------------------------------------------------------------------------


def test_private_chat_disabled_message_is_silent(plugin_class, tmp_path):
    plugin = _whitelist_plugin(
        plugin_class, tmp_path, enabled=False, whitelist=[], enable_private_chat=False
    )
    event = _private_event("hello")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


def test_private_chat_disabled_nju_command_is_silent(plugin_class, tmp_path):
    plugin = _whitelist_plugin(
        plugin_class, tmp_path, enabled=False, whitelist=[], enable_private_chat=False
    )
    event = _private_event("nju hello")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


def test_private_chat_disabled_nju_grep_does_not_run(
    plugin_class, tmp_path, monkeypatch
):
    plugin = _whitelist_plugin(
        plugin_class, tmp_path, enabled=False, whitelist=[], enable_private_chat=False
    )
    monkeypatch.setattr(
        GrepLocalDocsTool, "_run", _raise_if_called("async_GrepLocalDocsTool._run")
    )
    event = _private_event("nju_grep 关键词")
    results = asyncio.run(_collect(plugin.nju_grep(event, keywords="关键词")))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


def test_private_chat_disabled_nju_has_no_side_effects(
    plugin_class, tmp_path, monkeypatch
):
    plugin = _whitelist_plugin(
        plugin_class, tmp_path, enabled=False, whitelist=[], enable_private_chat=False
    )
    monkeypatch.setattr(
        plugin.agent, "answer", _raise_if_called("async_agent.answer")
    )
    monkeypatch.setattr(
        plugin.retriever, "search", _raise_if_called("async_retriever.search")
    )

    event = _private_event("nju hello")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
    assert not plugin.rate_limiter._buckets


def test_answer_question_directly_stops_unauthorized_event(
    plugin_class, tmp_path, monkeypatch
):
    plugin = _whitelist_plugin(plugin_class, tmp_path, enabled=True, whitelist=["g1"])
    monkeypatch.setattr(
        plugin.agent, "answer", _raise_if_called("async_agent.answer")
    )

    event = _group_event("南大助手 hello", group_id="g2")
    results = asyncio.run(_collect(plugin._answer_question(event, "hello")))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []


def test_answer_question_directly_stops_disabled_private_chat(
    plugin_class, tmp_path, monkeypatch
):
    plugin = _whitelist_plugin(
        plugin_class, tmp_path, enabled=False, whitelist=[], enable_private_chat=False
    )
    monkeypatch.setattr(
        plugin.agent, "answer", _raise_if_called("async_agent.answer")
    )

    event = _private_event("hello")
    results = asyncio.run(_collect(plugin._answer_question(event, "hello")))

    assert event.stopped is True
    assert results == []
    assert plugin.agent.calls == []
