import asyncio

import pytest

from nju_qa.agent import NO_EVIDENCE, NjuQaAgent
from nju_qa.doc_utils import parse_yuque_doc_url, read_document_content
from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.retriever import HybridRetriever


def make_index(tmp_path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    doc = Document(
        "1",
        "新生网站指南",
        "新生手册",
        "nju/guide",
        "freshman",
        "https://www.yuque.com/nju/guide/freshman",
        "a",
        "b",
        "新生请使用信息门户和教务网站。",
        tmp_path / "doc.md",
    )
    index.upsert(doc, [1.0, 0.0])
    return index, doc


def test_chinese_keywords_are_split_and_ranked(tmp_path):
    index, _ = make_index(tmp_path)
    assert index.keyword("新生需要看哪些网站", 10)[0][0]["title"] == "新生网站指南"


def test_read_doc_paginates_and_rejects_traversal(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntitle: t\n---\n\n|x|\n|-|\n|y|\n正文内容", encoding="utf-8"
    )
    assert read_document_content(tmp_path, "a.md", 0, 2)["has_more"]
    with pytest.raises(ValueError):
        read_document_content(tmp_path, "../a.md")


def test_yuque_url_query_and_anchor_parse():
    assert parse_yuque_doc_url("https://www.yuque.com/nju/guide/freshman?x=1#part") == (
        "nju/guide",
        "freshman",
    )


def test_keyword_fallback_without_embedding(tmp_path):
    index, _ = make_index(tmp_path)
    from nju_qa.config import PluginConfig

    report = asyncio.run(
        HybridRetriever(
            index,
            PluginConfig.from_mapping(
                {"yuque_repositories": ["nju/guide"], "score_threshold": 0}
            ),
        ).debug_search("新生网站")
    )
    assert report["mode"] == "keyword" and report["selected"]


class Event:
    unified_msg_origin = "u"


class Context:
    async def get_current_chat_provider_id(self, _):
        return "provider"


def test_no_evidence_hard_blocks_campus_fact():
    async def loop(**_):
        return type("R", (), {"completion_text": "根据一般经验请看官网"})()

    agent = NjuQaAgent(Context(), lambda _: [], loop)
    assert asyncio.run(agent.answer(Event(), "新生需要看哪些网站？")) == NO_EVIDENCE
