"""Tests for the in-memory sliding-window rate limiter."""

from __future__ import annotations

import asyncio

import pytest

from nju_qa.config import PluginConfig
from nju_qa.rate_limiter import RateLimiter
from tests.helpers import _FakeEvent, _collect, _make_plugin


def test_allows_up_to_max_count():
    limiter = RateLimiter(group_max=3, group_window_seconds=3600)
    for _ in range(3):
        allowed, state = limiter.is_allowed("g1", is_group=True)
        assert allowed is True
        assert state.current_count <= 3


def test_denies_after_max_count():
    limiter = RateLimiter(group_max=2, group_window_seconds=3600)
    limiter.is_allowed("g1", is_group=True)
    limiter.is_allowed("g1", is_group=True)
    allowed, state = limiter.is_allowed("g1", is_group=True)
    assert allowed is False
    assert state.silent is False
    assert state.current_count == 2
    assert state.max_count == 2
    assert state.reset_after_seconds > 0
    assert state.is_group is True


def test_subsequent_blocked_requests_are_silent():
    limiter = RateLimiter(group_max=1, group_window_seconds=3600)
    limiter.is_allowed("g1", is_group=True)
    _, first_block = limiter.is_allowed("g1", is_group=True)
    assert first_block.allowed is False and first_block.silent is False
    _, second_block = limiter.is_allowed("g1", is_group=True)
    assert second_block.allowed is False and second_block.silent is True


def test_reminder_resets_after_window(monkeypatch):
    limiter = RateLimiter(group_max=1, group_window_seconds=60)
    now = [0.0]
    monkeypatch.setattr("nju_qa.rate_limiter.time.monotonic", lambda: now[0])
    limiter.is_allowed("g1", is_group=True)
    now[0] = 30.0
    _, block = limiter.is_allowed("g1", is_group=True)
    assert block.silent is False
    now[0] = 61.0
    assert limiter.is_allowed("g1", is_group=True)[0] is True
    now[0] = 62.0
    _, block2 = limiter.is_allowed("g1", is_group=True)
    assert block2.allowed is False and block2.silent is False


def test_independent_tracking_per_key():
    limiter = RateLimiter(group_max=1, group_window_seconds=3600)
    assert limiter.is_allowed("g1", is_group=True)[0] is True
    assert limiter.is_allowed("g2", is_group=True)[0] is True
    assert limiter.is_allowed("g1", is_group=True)[0] is False
    assert limiter.is_allowed("g2", is_group=True)[0] is False


def test_private_and_group_are_independent():
    limiter = RateLimiter(
        group_max=1,
        group_window_seconds=3600,
        private_max=1,
        private_window_seconds=3600,
    )
    assert limiter.is_allowed("g1", is_group=True)[0] is True
    assert limiter.is_allowed("private:u1", is_group=False)[0] is True
    assert limiter.is_allowed("g1", is_group=True)[0] is False
    assert limiter.is_allowed("private:u1", is_group=False)[0] is False


def test_zero_max_disables_limiting():
    limiter = RateLimiter(group_max=0, group_window_seconds=3600)
    for _ in range(10):
        assert limiter.is_allowed("g1", is_group=True)[0] is True


def test_window_expires_old_entries(monkeypatch):
    limiter = RateLimiter(group_max=1, group_window_seconds=60)
    now = [0.0]
    monkeypatch.setattr(
        "nju_qa.rate_limiter.time.monotonic", lambda: now[0]
    )
    assert limiter.is_allowed("g1", is_group=True)[0] is True
    now[0] = 30.0
    assert limiter.is_allowed("g1", is_group=True)[0] is False
    now[0] = 61.0
    assert limiter.is_allowed("g1", is_group=True)[0] is True


def test_reset_clears_state():
    limiter = RateLimiter(group_max=1, group_window_seconds=3600)
    limiter.is_allowed("g1", is_group=True)
    limiter.reset()
    assert limiter.is_allowed("g1", is_group=True)[0] is True


def test_config_parses_rate_limit_fields():
    config = PluginConfig.from_mapping(
        {
            "yuque_repositories": ["nju/guide"],
            "group_rate_limit": 50,
            "group_rate_limit_window": 1800,
            "private_rate_limit": 10,
            "private_rate_limit_window": 7200,
        }
    )
    assert config.group_rate_limit == 50
    assert config.group_rate_limit_window == 1800
    assert config.private_rate_limit == 10
    assert config.private_rate_limit_window == 7200


def test_config_validates_rate_limit_ranges():
    with pytest.raises(ValueError):
        PluginConfig.from_mapping({"group_rate_limit": -1})
    with pytest.raises(ValueError):
        PluginConfig.from_mapping({"group_rate_limit": 2000})
    with pytest.raises(ValueError):
        PluginConfig.from_mapping({"group_rate_limit_window": 30})
    with pytest.raises(ValueError):
        PluginConfig.from_mapping({"private_rate_limit_window": 100000})


def test_group_handler_blocks_after_limit(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=2)
    event = _FakeEvent(group_id="g1", text="nju hello")

    first = asyncio.run(_collect(plugin.nju(event)))
    assert "答案：hello" in first[0]

    second = asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju hi"))))
    assert "答案：hi" in second[0]

    third = asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju hey"))))
    assert "本群已达到当前时段的提问上限" in third[0]
    assert "私聊" in third[0]

    # After the first reminder, further requests in the same window are silent.
    fourth = asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju again"))))
    assert fourth == []


def test_private_handler_blocks_after_limit(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, private_limit=2)

    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="", sender_id="u1", text="nju a"))))
    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="", sender_id="u1", text="nju b"))))
    third = asyncio.run(
        _collect(plugin.nju(_FakeEvent(group_id="", sender_id="u1", text="nju c")))
    )
    assert "你已达到当前时段的提问上限" in third[0]

    fourth = asyncio.run(
        _collect(plugin.nju(_FakeEvent(group_id="", sender_id="u1", text="nju d")))
    )
    assert fourth == []


def test_nju_grep_counts_toward_group_limit(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=1)
    # nju_grep does not need a real index for the rate-limit guard.
    event = _FakeEvent(group_id="g1", text="nju_grep 宿舍")
    asyncio.run(_collect(plugin.nju_grep(event)))
    # The first call is allowed; actual grep may fail because the index is empty,
    # but the handler still consumes one group quota.
    second = asyncio.run(
        _collect(plugin.nju(_FakeEvent(group_id="g1", text="nju hey")))
    )
    assert "本群已达到当前时段的提问上限" in second[0]


def test_admin_commands_are_not_rate_limited(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path, group_limit=1)
    # Consume the one group quota.
    asyncio.run(_collect(plugin.nju(_FakeEvent(group_id="g1", text="nju q"))))

    # Admin commands should still be allowed.
    sync_event = _FakeEvent(group_id="g1", text="nju_sync")
    result = asyncio.run(_collect(plugin.nju_sync(sync_event)))
    assert "已启动后台同步" in result[0]

    index_event = _FakeEvent(group_id="g1", text="nju_index rebuild")
    result = asyncio.run(_collect(plugin.nju_index(index_event, action="rebuild")))
    assert "已启动后台索引重建" in result[0]

    search_event = _FakeEvent(group_id="g1", text="nju_search test")
    result = asyncio.run(_collect(plugin.nju_search(search_event, query="test")))
    # nju_search may return "知识库为空" or a debug report; the point is it did not
    # hit the rate-limit message.
    assert "提问上限" not in result[0]

    debug_event = _FakeEvent(group_id="g1", text="nju_debug")
    result = asyncio.run(_collect(plugin.nju_debug(debug_event)))
    assert "handler_arg" in result[0]
    assert "matched_source" in result[0]


def test_nju_handler_ignores_other_commands(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(group_id="g1", text="audit ok")
    result = asyncio.run(_collect(plugin.nju(event)))
    assert result == []
    assert not event.stopped


def test_nju_grep_handler_ignores_other_commands(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(group_id="g1", text="nju source 宿舍")
    result = asyncio.run(_collect(plugin.nju_grep(event)))
    assert result == []
    assert not event.stopped


def test_admin_handlers_ignores_other_commands(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)

    sync_event = _FakeEvent(group_id="g1", text="nju_index rebuild")
    assert asyncio.run(_collect(plugin.nju_sync(sync_event))) == []
    assert not sync_event.stopped

    index_event = _FakeEvent(group_id="g1", text="nju_search test")
    assert asyncio.run(_collect(plugin.nju_index(index_event))) == []
    assert not index_event.stopped

    search_event = _FakeEvent(group_id="g1", text="nju_debug")
    assert asyncio.run(_collect(plugin.nju_search(search_event))) == []
    assert not search_event.stopped

    debug_event = _FakeEvent(group_id="g1", text="nju_sync")
    assert asyncio.run(_collect(plugin.nju_debug(debug_event))) == []
    assert not debug_event.stopped


def test_nju_handler_does_not_swallow_nju_grep(plugin_class, tmp_path):
    plugin = _make_plugin(plugin_class, tmp_path)
    event = _FakeEvent(group_id="g1", text="nju_grep 宿舍")
    result = asyncio.run(_collect(plugin.nju(event)))
    assert result == []
    assert not event.stopped
