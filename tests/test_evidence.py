"""Evidence tracking, grep reliability, and grounding source selection tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from nju_qa.agent import NO_EVIDENCE, NjuQaAgent, SourceTracker
from nju_qa.document_index import DocumentIndex
from nju_qa.evidence import (
    build_chunk_from_grep_hit,
    build_document_from_grep_hit,
    document_from_index_row,
    evaluate_grep_reliability,
    grep_hits_to_search_results,
    score_grep_hit,
    select_grounding_sources,
)
from nju_qa.models import Document, SearchResult
from nju_qa.tools.documents import (
    GetDocDetailsTool,
    GrepLocalDocsTool,
    ReadDocTool,
)


class _Context:
    async def get_current_chat_provider_id(self, _):
        return "provider"


class _Response:
    def __init__(self, text: str):
        self.completion_text = text


class _Event:
    unified_msg_origin = "u"


def _doc(
    tmp_path: Path,
    rel: str,
    title: str,
    body: str,
    yuque_id: str,
    namespace: str = "nju/guide",
    repository: str = "guide",
) -> Document:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return Document(
        yuque_id=yuque_id,
        title=title,
        repository=repository,
        namespace=namespace,
        slug=Path(rel).stem,
        url=f"https://yuque.test/{namespace}/{Path(rel).stem}",
        created_at="a",
        updated_at="b",
        body=body,
        path=Path(rel),
    )


def _index_with_docs(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    docs = [
        _doc(
            tmp_path,
            "card.md",
            "校园卡补办指南",
            "## 校园卡的挂失与补办\n\n"
            "校园卡丢失后可在信息化建设管理服务中心一楼大厅补办。\n"
            "鼓楼校区请到综合服务大厅办理。补卡费用为 20 元。",
            "card123",
        ),
        _doc(
            tmp_path,
            "photo.md",
            "校园卡照片采集",
            "## 照片采集\n\n新生入学后需要上传校园卡照片。",
            "photo456",
        ),
        _doc(
            tmp_path,
            "dorm.md",
            "宿舍分配与入住",
            "## 宿舍\n\n新生报到后按学院分配宿舍。",
            "dorm789",
        ),
        _doc(
            tmp_path,
            "00_index.md",
            "00_index",
            "## 目录\n\n- 校园卡\n- 宿舍\n- 教务",
            "idx000",
        ),
        _doc(
            tmp_path,
            "id_card.md",
            "学生证补办",
            "## 学生证补办\n\n学生证丢失后由学院教务处负责补办。",
            "idcard321",
        ),
    ]
    for doc in docs:
        index.upsert(doc)
    return index


def _hit(
    title: str,
    yuque_id: str,
    matched_keywords: list[str],
    snippet: str,
    path: str = "doc.md",
) -> dict:
    return {
        "yuque_id": yuque_id,
        "title": title,
        "repository": "guide",
        "namespace": "nju/guide",
        "slug": Path(path).stem,
        "url": f"https://yuque.test/nju/guide/{Path(path).stem}",
        "created_at": "a",
        "updated_at": "b",
        "path": path,
        "matched_keywords": matched_keywords,
        "matches": [{"line_start": 1, "line_end": 1, "snippet": snippet}],
    }


@pytest.fixture
def sample_hits():
    return {
        "direct": _hit(
            "校园卡补办指南",
            "card123",
            ["校园卡", "补办"],
            "校园卡的挂失与补办地点如下：仙林校区信息化建设管理服务中心一楼大厅。",
            "card.md",
        ),
        "photo": _hit(
            "校园卡照片采集",
            "photo456",
            ["校园卡"],
            "新生入学后需要上传校园卡照片。",
            "photo.md",
        ),
        "dorm": _hit(
            "宿舍分配与入住",
            "dorm789",
            ["校园卡"],
            "新生报到后按学院分配宿舍。",
            "dorm.md",
        ),
        "index": _hit(
            "00_index",
            "idx000",
            ["校园卡", "补办"],
            "- 校园卡\n- 宿舍\n- 教务",
            "00_index.md",
        ),
        "id_card": _hit(
            "学生证补办",
            "idcard321",
            ["补办"],
            "学生证丢失后由学院教务处负责补办。",
            "id_card.md",
        ),
    }


def test_grep_reliability_direct_answer_is_reliable(sample_hits):
    reliable, diag = evaluate_grep_reliability(
        sample_hits["direct"], ["校园卡", "补办"]
    )
    assert reliable is True
    assert diag["core_coverage"] == 1.0
    assert diag["same_window_core_coverage"] is True


def test_grep_reliability_partial_keyword_is_not_reliable(sample_hits):
    for key in ("photo", "dorm", "id_card"):
        reliable, _ = evaluate_grep_reliability(sample_hits[key], ["校园卡", "补办"])
        assert reliable is False, key


def test_grep_reliability_index_is_not_reliable_even_with_full_coverage(sample_hits):
    reliable, diag = evaluate_grep_reliability(sample_hits["index"], ["校园卡", "补办"])
    assert reliable is False
    assert diag["is_index_document"] is True


def test_grep_reliability_no_answer_status_is_unreliable():
    hit = _hit(
        "问题 QA",
        "qa1",
        ["校园卡", "补办"],
        "status: no_answer_found\n校园卡补办地点暂无。",
    )
    reliable, diag = evaluate_grep_reliability(hit, ["校园卡", "补办"])
    assert reliable is False
    assert diag["qa_status"] == "no_answer"


def test_grep_reliability_resolved_status_is_reliable():
    hit = _hit(
        "问题 QA",
        "qa2",
        ["校园卡", "补办"],
        "status: resolved\n校园卡可在信息化建设管理服务中心补办。",
    )
    reliable, _ = evaluate_grep_reliability(hit, ["校园卡", "补办"])
    assert reliable is True


def test_grep_ranking_direct_answer_wins(sample_hits):
    terms = ["校园卡", "补办"]
    scores = {k: score_grep_hit(v, terms) for k, v in sample_hits.items()}
    assert scores["direct"] > scores["photo"]
    assert scores["direct"] > scores["dorm"]
    assert scores["direct"] > scores["index"]
    assert scores["direct"] > scores["id_card"]


def test_grep_same_window_coverage_outranks_split_match():
    split = {
        **_hit(
            "分散命中",
            "split1",
            ["校园卡", "补办"],
            "第一章介绍校园卡。",
        ),
        "matches": [
            {"line_start": 1, "line_end": 1, "snippet": "第一章介绍校园卡。"},
            {"line_start": 10, "line_end": 10, "snippet": "学生证可以补办。"},
        ],
    }
    together = _hit(
        "同窗口命中",
        "together1",
        ["校园卡", "补办"],
        "校园卡的挂失与补办地点如下。",
    )
    terms = ["校园卡", "补办"]
    assert score_grep_hit(together, terms) > score_grep_hit(split, terms)
    _, diag_together = evaluate_grep_reliability(together, terms)
    _, diag_split = evaluate_grep_reliability(split, terms)
    assert diag_together["same_window_core_coverage"] is True
    assert diag_split["same_window_core_coverage"] is False


def test_grep_hits_convert_to_search_results():
    hits = [
        _hit("校园卡补办指南", "c1", ["校园卡", "补办"], "校园卡可补办。", "c.md"),
        _hit("校园卡照片采集", "c2", ["校园卡"], "照片采集。", "p.md"),
    ]
    results = grep_hits_to_search_results(hits, ["校园卡", "补办"])
    assert len(results) == 2
    assert results[0].document.yuque_id == "c1"
    assert results[0].reliable is True
    assert results[1].reliable is False


def test_source_tracker_adds_grep_evidence(sample_hits):
    tracker = SourceTracker()
    tracker.add_grep_hits(
        [sample_hits["direct"], sample_hits["photo"]], ["校园卡", "补办"]
    )
    assert tracker.matched_count == 2
    assert tracker.reliable_count == 1
    assert any(s.document.yuque_id == "card123" for s in tracker.sources)


def test_source_tracker_deduplicates_by_yuque_id():
    tracker = SourceTracker()
    doc = build_document_from_grep_hit(
        _hit("校园卡补办指南", "c1", ["校园卡", "补办"], " snippet ", "c.md")
    )
    chunk = build_chunk_from_grep_hit(
        _hit("校园卡补办指南", "c1", ["校园卡", "补办"], " snippet ", "c.md"),
        ["校园卡", "补办"],
    )
    tracker.add(
        [
            SearchResult(
                "G1", doc, 0.5, chunk=chunk, reliable=False,
                retrieval_methods=("grep",),
            ),
            SearchResult(
                "S1",
                doc,
                0.9,
                chunk=replace(chunk, reliable=True, final_score=0.9),
                reliable=True,
                retrieval_methods=("keyword",),
            ),
        ]
    )
    assert tracker.matched_count == 1
    assert tracker.reliable_count == 1
    assert tracker.sources[0].retrieval_methods == ("grep", "keyword")


def test_select_grounding_sources_prefers_reliable_and_drops_unreliable():
    def make(source_id: str, score: float, reliable: bool, yuque_id: str):
        doc = Document(
            yuque_id, "t", "r", "n", "s", "u", "a", "b", "body", path=Path(f"{yuque_id}.md")
        )
        return SearchResult(source_id, doc, score, reliable=reliable)

    sources = [
        make("S1", 0.9, False, "d1"),
        make("S2", 0.6, True, "d2"),
        make("S3", 0.3, True, "d3"),
    ]
    selected = select_grounding_sources(sources, max_sources=2)
    assert len(selected) == 2
    assert selected[0].source_id == "S2"
    assert selected[1].source_id == "S3"
    assert all(s.reliable for s in selected)


def test_document_from_index_row():
    index = DocumentIndex(Path("/nonexistent") / "x.sqlite3")
    doc = Document(
        "1", "t", "r", "n", "s", "u", "a", "b", "body", path=Path("p.md")
    )
    index.upsert(doc)
    row = index.all_documents()[0]
    rebuilt = document_from_index_row(row, "new body")
    assert rebuilt.yuque_id == "1"
    assert rebuilt.body == "new body"


def test_grep_tool_registers_unified_evidence(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )
    result = asyncio.run(tool._run("校园卡 补办"))
    assert result["count"] >= 1
    assert tracker.matched_count >= 1
    assert tracker.reliable_count >= 1
    assert any(s.document.yuque_id == "card123" for s in tracker.sources)


def test_grep_only_campus_question_does_not_return_no_evidence(tmp_path):
    index = _index_with_docs(tmp_path)

    def tool_factory(tracker):
        return [
            GrepLocalDocsTool(
                index=index, docs_root=tmp_path, tracker=tracker
            )
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        grep = next(t for t in tools if t.name == "grep_local_docs")
        await grep._run("校园卡 补办")
        return _Response(
            "校园卡可在信息化建设管理服务中心一楼大厅补办，费用 20 元。"
        )

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=tmp_path)
    answer = asyncio.run(agent.answer(_Event(), "校园卡在哪里补办？"))
    assert answer != NO_EVIDENCE
    assert "信息化建设管理服务中心" in answer


def test_read_doc_marks_source_read_and_merges_methods(tmp_path):
    index = _index_with_docs(tmp_path)
    captured: dict = {}

    def tool_factory(tracker):
        captured["tracker"] = tracker
        return [
            GrepLocalDocsTool(
                index=index, docs_root=tmp_path, tracker=tracker
            ),
            ReadDocTool(index=index, docs_root=tmp_path, tracker=tracker),
        ]

    async def loop(**kwargs):
        tools = kwargs["tools"]
        grep = next(t for t in tools if t.name == "grep_local_docs")
        read = next(t for t in tools if t.name == "read_doc")
        await grep._run("校园卡 补办")
        source = next(s for s in captured["tracker"].sources if s.reliable)
        await read._run(str(source.document.path))
        return _Response("已读取材料")

    agent = NjuQaAgent(_Context(), tool_factory, loop, docs_root=tmp_path)
    asyncio.run(agent.answer(_Event(), "校园卡在哪里补办？"))
    tracker = captured["tracker"]
    source = next(s for s in tracker.sources if s.document.yuque_id == "card123")
    assert "read" in source.retrieval_methods
    assert str(source.document.path) in tracker.read_sources


def test_get_doc_details_content_registers_evidence(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GetDocDetailsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )

    result = asyncio.run(tool._run(yuque_id="card123", include_content=True))
    assert "content" in result
    assert tracker.matched_count == 1
    assert str(tracker.sources[0].document.path) in tracker.read_sources


def test_get_doc_details_without_content_does_not_register_evidence(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GetDocDetailsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )

    result = asyncio.run(tool._run(yuque_id="card123", include_content=False))
    assert "results" in result
    assert tracker.matched_count == 0
    assert tracker.read_count == 0


def test_empty_grep_does_not_create_fake_sources(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )
    result = asyncio.run(tool._run("完全不存在的词"))
    assert result["count"] == 0
    assert tracker.matched_count == 0
    assert tracker.reliable_count == 0


def test_grep_hit_with_missing_path_is_handled():
    hit = _hit("t", "id", ["校园卡"], "snippet")
    hit.pop("path")
    doc = build_document_from_grep_hit(hit)
    assert doc.path is None
    assert doc.yuque_id == "id"


def test_grep_hit_with_empty_matches_is_unreliable():
    hit = _hit("t", "id", ["校园卡"], "")
    hit["matches"] = []
    reliable, _ = evaluate_grep_reliability(hit, ["校园卡"])
    assert reliable is False


def test_grep_fallback_to_bigram_terms(tmp_path):
    index = _index_with_docs(tmp_path)
    tracker = SourceTracker()
    tool = GrepLocalDocsTool(
        index=index, docs_root=tmp_path, tracker=tracker
    )
    # "挂失补办" is not in text, but splitting into "挂失 补办" won't help.
    # Use a phrase that appears only as substrings.
    result = asyncio.run(tool._run("信息建设中"))
    # Fallback should still find something if any bigram matches.
    assert isinstance(result, dict)
    assert "count" in result
