"""Security regression tests for message routing and default-LLM bypass."""

from __future__ import annotations

import asyncio

import pytest

from tests.helpers import _FakeEvent, _collect, _make_plugin


class CommandFilter:
    pass


class _FakeCommandHandler:
    def __init__(self, full_name: str) -> None:
        self.handler_full_name = full_name
        self.event_filters = [CommandFilter()]


class _FakeAllMessageHandler:
    def __init__(self, full_name: str) -> None:
        self.handler_full_name = full_name
        self.event_filters = []


def _at_chain(text: str):
    from astrbot.api import message_components as comp

    return [comp.At("bot"), comp.Plain(text)]


def _slash_event(text: str, raw_text: str | None = None, is_command: bool = True):
    """Build a fake event for an unmatched ``/`` wake-prefix message."""
    return _FakeEvent(
        group_id="g1",
        text=text,
        raw_text=raw_text if raw_text is not None else f"/{text}",
        is_command=is_command,
        is_at_or_wake_command=True,
    )


@pytest.mark.parametrize("is_command", [True, False])
def test_unknown_slash_suppresses_llm(is_command, plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("你好", is_command=is_command)
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == [True]
    assert event.call_llm is True
    assert plugin.agent.calls == []


def test_unknown_slash_with_prefix_stripped_from_chain_suppresses_llm(
    plugin_class, tmp_path,
):
    """Adapter may strip '/' from message_obj; fallback uses is_at_or_wake_command."""
    plugin = _make_plugin(plugin_class, tmp_path)
    from astrbot.api import message_components as comp

    event = _FakeEvent(
        group_id="g1",
        text="你好",
        raw_text="/你好",
        is_command=True,
        is_at_or_wake_command=True,
        message_chain=[comp.Plain("你好")],
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == [True]
    assert plugin.agent.calls == []


def test_other_plugin_command_match_is_passed_through(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("audit list", raw_text="/audit list", is_command=True)
    event.set_extra(
        "handlers_parsed_params",
        {"other_plugin.main.OtherPlugin.audit": {"args": ["list"]}},
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == []
    assert plugin.agent.calls == []


def test_other_plugin_command_match_no_stop_is_passed_through(plugin_class, tmp_path):
    """A registered command whose handler did not stop the event must not trigger NJU QA."""
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("sid", raw_text="/sid", is_command=True)
    event.set_extra(
        "activated_handlers",
        [_FakeCommandHandler("other_plugin.main.OtherPlugin.sid")],
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == []
    assert plugin.agent.calls == []


def test_is_command_false_but_handlers_parsed_params_match_is_passed_through(
    plugin_class, tmp_path,
):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("sid", raw_text="/sid", is_command=False)
    event.set_extra(
        "handlers_parsed_params",
        {"other_plugin.main.OtherPlugin.sid": {}},
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == []
    assert plugin.agent.calls == []


def test_other_plugin_all_message_handler_can_process_unknown_slash(
    plugin_class, tmp_path,
):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("translate hello", raw_text="/translate hello")
    event.set_extra(
        "activated_handlers",
        [_FakeAllMessageHandler("translate_plugin.main.TranslatePlugin.on_message")],
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    # Unknown slash still suppresses default LLM, but the event is not stopped so
    # the other ALL-message handler can process it.
    assert event.should_call_llm_calls == [True]
    assert plugin.agent.calls == []


def test_nju_own_command_match_is_passed_through(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("nju_sync status", raw_text="/nju_sync status")
    event.set_extra(
        "handlers_parsed_params",
        {"astrbot_plugin_nju_qa.main.NjuQaPlugin.nju_sync": {"action": "status"}},
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    # on_message sees a registered command match and lets the command handler run.
    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == []
    assert plugin.agent.calls == []


def test_nju_command_handler_stops_event_before_on_message(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("nju hello", raw_text="/nju hello")
    event.stopped = True
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert results == []
    assert event.stopped is True
    assert plugin.agent.calls == []


def test_unknown_slash_does_not_consume_rate_limit(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=1)
    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju q"))))

    event = _slash_event("继续回答", raw_text="/继续回答")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == [True]
    assert plugin.agent.calls == ["q"]


def test_nju_command_uses_shared_qa_entry(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(group_id="g1", text="nju hello")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_at_mention_uses_shared_qa_entry(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(
        group_id="g1",
        text="hello",
        raw_text="hello",
        is_at_or_wake_command=True,
        message_chain=_at_chain("hello"),
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_wake_word_uses_shared_qa_entry(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(
        group_id="g1",
        text="南大助手 hello",
        raw_text="南大助手 hello",
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_private_chat_uses_shared_qa_entry(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(
        group_id="",
        sender_id="u1",
        text="hello",
        raw_text="hello",
        is_at_or_wake_command=True,
    )
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "答案：hello" in results[0]
    assert plugin.agent.calls == ["hello"]


def test_at_and_nju_share_rate_limit_key(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=2)

    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju a"))))
    at_event = _FakeEvent(
        group_id="g1",
        text="b",
        raw_text="b",
        is_at_or_wake_command=True,
        message_chain=_at_chain("b"),
    )
    asyncio.run(_collect(plugin.on_message(at_event)))

    third = _FakeEvent(group_id="g1", text="nju c")
    results = asyncio.run(_collect(plugin.nju(third)))
    assert "本群已达到当前时段的提问上限" in results[0]
    assert plugin.agent.calls == ["a", "b"]


def test_rate_limited_at_does_not_fall_back_to_default_llm(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=1)
    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju q"))))

    at_event = _FakeEvent(
        group_id="g1",
        text="again",
        raw_text="again",
        is_at_or_wake_command=True,
        message_chain=_at_chain("again"),
    )
    results = asyncio.run(_collect(plugin.on_message(at_event)))

    assert at_event.stopped is True
    assert len(results) == 1
    assert "本群已达到当前时段的提问上限" in results[0]
    assert plugin.agent.calls == ["q"]

    at_event2 = _FakeEvent(
        group_id="g1",
        text="again2",
        raw_text="again2",
        is_at_or_wake_command=True,
        message_chain=_at_chain("again2"),
    )
    results2 = asyncio.run(_collect(plugin.on_message(at_event2)))
    assert at_event2.stopped is True
    assert results2 == []
    assert plugin.agent.calls == ["q"]


def test_typo_njuu_silently_suppresses_llm(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("njuu 你好", raw_text="/njuu 你好")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == [True]
    assert plugin.agent.calls == []


def test_bare_slash_silently_suppresses_llm(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _slash_event("", raw_text="/")
    results = asyncio.run(_collect(plugin.on_message(event)))

    assert event.stopped is False
    assert results == []
    assert event.should_call_llm_calls == [True]
    assert plugin.agent.calls == []


def test_nju_help_and_source_do_not_count_as_qa(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=1)
    help_event = _FakeEvent(group_id="g1", text="nju help")
    asyncio.run(_collect(plugin.nju(help_event)))
    assert plugin.agent.calls == []

    source_event = _FakeEvent(group_id="g1", text="nju source 宿舍")
    asyncio.run(_collect(plugin.nju(source_event)))
    assert plugin.agent.calls == []

    qa_event = _FakeEvent(group_id="g1", text="nju real")
    results = asyncio.run(_collect(plugin.nju(qa_event)))
    assert "答案：real" in results[0]
    assert plugin.agent.calls == ["real"]


def test_nju_command_rate_limit_silent_block_stops_event(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=1)
    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju q"))))

    event = _FakeEvent(group_id="g1", text="nju blocked")
    results = asyncio.run(_collect(plugin.nju(event)))

    assert event.stopped is True
    assert len(results) == 1
    assert "本群已达到当前时段的提问上限" in results[0]
    assert plugin.agent.calls == ["q"]

    event2 = _FakeEvent(group_id="g1", text="nju blocked2")
    results2 = asyncio.run(_collect(plugin.nju(event2)))
    assert event2.stopped is True
    assert results2 == []
    assert plugin.agent.calls == ["q"]
