import asyncio
from dataclasses import dataclass
from pathlib import Path

from nju_qa.agent import NO_PROVIDER, NjuQaAgent, SourceTracker
from nju_qa.evidence import evidence_excerpt_from_text
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


def test_small_talk_uses_one_llm_call_without_search():
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        return Response("你好！我是由 NOVA 开发的南京大学校园问答助手。")

    agent = NjuQaAgent(
        Context(), lambda tracker: [], loop
    )
    answer = asyncio.run(agent.answer(Event(), "你好"))
    assert answer.startswith("你好")
    assert len(calls) == 1
    assert "RESEARCH" not in str(calls[0].get("system_prompt", ""))


def test_no_provider_is_friendly_and_does_not_run_agent():
    agent = NjuQaAgent(Context(None), lambda tracker: [], None)
    assert asyncio.run(agent.answer(Event(), "你好")) == NO_PROVIDER


def test_citations_are_deduplicated_from_actual_evidence():
    tracker = SourceTracker()
    tracker.add_read_document(
        Document(
            "1", "t", "r", "n", "s", "https://yuque.test/doc",
            "a", "b", "body", path=Path("doc.md"),
        ),
        "content",
    )
    # Adding the same document again should not duplicate read_sources.
    tracker.add_read_document(
        Document(
            "1", "t", "r", "n", "s", "https://yuque.test/doc",
            "a", "b", "body", path=Path("doc.md"),
        ),
        "content",
    )
    assert len(tracker.evidence_excerpts) == 2
    assert len(tracker.read_sources) == 1


def test_verified_url_is_recorded_from_read_content():
    tracker = SourceTracker()
    tracker.add_evidence(
        evidence_excerpt_from_text(
            "官方入口：https://portal.nju.edu.cn",
            title="t",
            file_path="doc.md",
        )
    )
    assert "https://portal.nju.edu.cn" in tracker.verified_urls


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


def test_search_knowledge_base_registers_candidates():
    retriever = Retriever([source()])
    tracker = SourceTracker()
    output = asyncio.run(search_knowledge_base(retriever, tracker, "通识课学分"))
    assert retriever.queries == ["通识课学分"]
    assert len(tracker.candidate_sources) == 1
    assert tracker.candidate_sources[0].document.yuque_id == "1"
    assert output["candidates"][0]["source_url"] == "https://yuque.test/doc"


def test_unsynced_knowledge_base_tells_user_to_contact_admin():
    output = asyncio.run(
        search_knowledge_base(Retriever([], rows=False), SourceTracker(), "校园卡")
    )
    assert "/nju_sync" in output["reason"]


def test_fact_question_without_read_evidence_returns_no_evidence():
    async def loop(**kwargs):
        # Research phase returns text but never reads a document.
        return Response("根据经验，校园卡可在中心补办。")

    agent = NjuQaAgent(Context(), lambda tracker: [], loop)
    answer = asyncio.run(agent.answer(Event(), "校园卡在哪里补办？"))
    assert "知识库中暂未找到可靠资料" in answer


def test_answer_without_evidence_markers_is_rejected():
    doc = Document(
        "1", "t", "r", "n", "s", "https://yuque.test/doc",
        "a", "b", "body", path=Path("doc.md"),
    )

    async def loop(**kwargs):
        if kwargs.get("system_prompt", "").startswith("你是南京大学校园问答助手"):
            return Response("校园卡可在中心补办")  # missing [E#]
        return Response("ignored")

    agent = NjuQaAgent(Context(), lambda tracker: [], loop)
    tracker = SourceTracker()
    tracker.add_read_document(doc, "校园卡可在中心补办")
    # Patch agent's answer phase by injecting the tracker manually.
    answer = asyncio.run(agent._answer_phase(Event(), "校园卡在哪里补办？", tracker))
    assert "知识库中暂未找到可靠资料" in answer or "无法" in answer
