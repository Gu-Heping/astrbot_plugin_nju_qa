import asyncio
from dataclasses import dataclass
from pathlib import Path

from nju_qa.agent import NO_PROVIDER, NjuQaAgent, SourceTracker
from nju_qa.knowledge_tools import search_knowledge_base
from nju_qa.models import Document, SearchResult
from nju_qa.routing import MessageRouter, mark_command_handled


class Event:
    def __init__(self, group_id=None, sender_id="tester"):
        self.group_id = group_id
        self.sender_id = sender_id
        self.unified_msg_origin = "test:session"
        self.stop_calls = 0

    def get_group_id(self):
        return self.group_id

    def get_sender_id(self):
        return self.sender_id

    def stop_event(self):
        self.stop_calls += 1


def source() -> SearchResult:
    document = Document(
        "1",
        "通识课说明",
        "本科生指南",
        "nju/guide",
        "general",
        "https://yuque.test/doc",
        "2026-01-01",
        "2026-02-01",
        "通识课要求",
        Path("doc.md"),
    )
    return SearchResult("S1", document, 1.0, reliable=True)


def test_commands_never_enter_general_router():
    router = MessageRouter(("nju",), True, True)
    event = Event()
    assert not router.route(event, "/nju_sync", False).should_handle
    assert not router.route(event, "/nju_sync status", False).should_handle
    mark_command_handled(event)
    assert event.stop_calls == 1
    assert not router.route(event, "hello", False).should_handle


def test_private_and_group_route_once_only_when_explicit():
    router = MessageRouter(("nju",), True, True)
    assert router.route(Event(), "你好", False).query == "你好"
    assert not router.route(Event("g"), "你好", False).should_handle
    assert router.route(Event("g"), "你好", True).query == "你好"
    assert router.route(Event("g"), "nju，你好", False).query == "你好"
    # Private chat may report empty string instead of None for group_id.
    assert router.route(Event(group_id=""), "你好", False).query == "你好"


@dataclass
class Response:
    completion_text: str


class Context:
    def __init__(self, provider="provider"):
        self.provider = provider

    async def get_current_chat_provider_id(self, _umo):
        return self.provider


def test_normal_chat_uses_one_llm_call_without_forcing_search():
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        return Response("你好！我是南京大学校园问答助手。")

    agent = NjuQaAgent(
        Context(), lambda tracker: [type("Tool", (), {"tracker": tracker})()], loop
    )
    answer = asyncio.run(agent.answer(Event(), "你好"))
    assert answer.startswith("你好")
    assert len(calls) == 1


def test_fact_answer_can_only_cite_tool_tracked_sources():
    async def loop(**kwargs):
        kwargs["tools"][0].tracker.add([source()])
        kwargs["tools"][0].tracker.read_sources.add("doc.md")
        return Response(
            "通识课要求见资料：https://fake.test\n参考来源：\n1. 《伪造》：https://fake.test"
        )

    agent = NjuQaAgent(
        Context(), lambda tracker: [type("Tool", (), {"tracker": tracker})()], loop
    )
    answer = asyncio.run(agent.answer(Event(), "通识课需要多少学分？"))
    assert "https://yuque.test/doc" in answer
    assert "https://fake.test" not in answer


def test_no_provider_is_friendly_and_does_not_run_agent():
    agent = NjuQaAgent(Context(None), lambda tracker: [], None)
    assert asyncio.run(agent.answer(Event(), "你好")) == NO_PROVIDER


def test_fact_search_without_agent_read_retries_with_selected_document_body():
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["tools"][0].tracker.add([source()])
        return Response("基于已读材料的回答")

    agent = NjuQaAgent(
        Context(), lambda tracker: [type("Tool", (), {"tracker": tracker})()], loop
    )
    answer = asyncio.run(agent.answer(Event(), "通识课需要多少学分？"))
    assert len(calls) == 2 and "已读材料" in calls[1]["prompt"]
    assert "https://yuque.test/doc" in answer


def test_no_tool_results_leave_no_fake_citation():
    async def loop(**kwargs):
        return Response("知识库中暂未找到可靠答案")

    agent = NjuQaAgent(Context(), lambda tracker: [], loop)
    assert "参考来源" not in asyncio.run(agent.answer(Event(), "校园卡在哪里补办？"))


def test_citations_are_deduplicated_from_actual_tool_results():
    tracker = SourceTracker()
    tracker.add([source(), source()])
    assert len(tracker.sources) == 1


def test_verified_website_url_from_read_document_is_preserved():
    tracker = SourceTracker()
    tracker.record_read_content("官方入口：https://portal.nju.edu.cn")
    from nju_qa.agent import append_verified_citations

    answer = append_verified_citations(
        "请访问 https://portal.nju.edu.cn；不要访问 https://fake.test",
        [],
        tracker.verified_urls,
    )
    assert "https://portal.nju.edu.cn" in answer
    assert "https://fake.test" not in answer


class Index:
    def __init__(self, rows):
        self.rows = rows

    def all_documents(self):
        return self.rows


class Retriever:
    def __init__(self, results, rows=True):
        self.results = results
        self.index = Index([object()] if rows else [])
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.results


def test_campus_question_searches_tool_and_returns_only_its_source():
    retriever = Retriever([source()])
    tracker = SourceTracker()
    output = asyncio.run(search_knowledge_base(retriever, tracker, "通识课学分"))
    assert retriever.queries == ["通识课学分"]
    assert tracker.sources == [source()]
    assert output["candidates"][0]["source_url"] == "https://yuque.test/doc"


def test_unsynced_knowledge_base_tells_user_to_contact_admin():
    output = asyncio.run(
        search_knowledge_base(Retriever([], rows=False), SourceTracker(), "校园卡")
    )
    assert "/nju_sync" in output["reason"]
