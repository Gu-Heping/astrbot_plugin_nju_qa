"""Tests for the code-level retrieval plan executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from nju_qa.agent import SourceTracker
from nju_qa.chunk_store import ChunkStore
from nju_qa.chunking import Chunk, _content_hash, _stable_chunk_id
from nju_qa.config import PluginConfig
from nju_qa.document_index import DocumentIndex
from nju_qa.models import Document
from nju_qa.retrieval_executor import RetrievalExecutor
from nju_qa.retrieval_plan import build_retrieval_plan
from nju_qa.retriever import HybridRetriever


def _chunk(
    document_id: str,
    chunk_index: int,
    content: str,
    *,
    title: str = "",
    repository: str = "repo",
    namespace: str = "QA",
    slug: str = "",
    file_path: str = "",
) -> Chunk:
    return Chunk(
        chunk_id=_stable_chunk_id(document_id, chunk_index, content),
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        content_hash=_content_hash(content),
        title=title,
        repository=repository,
        namespace=namespace,
        slug=slug or document_id,
        file_path=file_path or f"{namespace}/{document_id}.md",
        source_url="",
        updated_at="b",
    )


def _doc(
    tmp_path: Path,
    rel: str,
    title: str,
    body: str,
    yuque_id: str,
    namespace: str = "QA",
    repository: str = "repo",
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


@pytest.fixture
def executor_setup(tmp_path: Path):
    index = DocumentIndex(tmp_path / "index.sqlite3")
    docs = [
        _doc(
            tmp_path,
            "QA/01_入学与行政事务/校园卡.md",
            "校园卡",
            "校园卡可在信息化建设管理服务中心补办。",
            "card",
        ),
        _doc(
            tmp_path,
            "QA/02_教务与学业/培养方案/开甲学院.md",
            "开甲学院培养方案",
            "开甲学院大一学习数学、程序设计。",
            "kaijia",
        ),
        _doc(
            tmp_path,
            "QA/02_教务与学业/选课.md",
            "选课",
            "选课系统在教务系统开放。",
            "course",
        ),
    ]
    for doc in docs:
        index.upsert(doc)

    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.save_document_chunks(
        "card",
        [
            _chunk(
                "card",
                0,
                "校园卡可在信息化建设管理服务中心补办。",
                title="校园卡",
                file_path="QA/01_入学与行政事务/校园卡.md",
            ),
        ],
    )
    chunk_store.save_document_chunks(
        "kaijia",
        [
            _chunk(
                "kaijia",
                0,
                "开甲学院大一学习数学、程序设计。",
                title="开甲学院培养方案",
                file_path="QA/02_教务与学业/培养方案/开甲学院.md",
            ),
        ],
    )
    chunk_store.save_document_chunks(
        "course",
        [
            _chunk(
                "course",
                0,
                "选课系统在教务系统开放。",
                title="选课",
                file_path="QA/02_教务与学业/选课.md",
            ),
        ],
    )

    config = PluginConfig.from_mapping(
        {"enable_vector_search": False, "retrieval_top_k": 5, "score_threshold": 0.1}
    )
    retriever = HybridRetriever(index, config, chunk_store=chunk_store)
    return RetrievalExecutor(retriever, index, tmp_path), index, tmp_path


@pytest.mark.asyncio
async def test_executor_finds_direct_entity_evidence(executor_setup):
    executor, index, _ = executor_setup
    plan = build_retrieval_plan("开甲学院大一要学什么", index.all_documents())
    tracker = SourceTracker()
    result = await executor.execute(plan, tracker)

    assert result.has_direct
    assert any(
        c.status.value == "DIRECT" and c.need.entity_terms
        for c in result.coverage
    )
    assert tracker.reliable_count >= 1


@pytest.mark.asyncio
async def test_executor_reports_zero_entity_hit(executor_setup):
    executor, index, _ = executor_setup
    plan = build_retrieval_plan("火星学院怎么进", index.all_documents())
    tracker = SourceTracker()
    result = await executor.execute(plan, tracker)

    assert len(result.zero_entity_hits) == 1
    assert result.zero_entity_hits[0].entity_terms == ["火星学院"]


@pytest.mark.asyncio
async def test_executor_runs_background_for_unknown_entity(executor_setup):
    executor, index, _ = executor_setup
    plan = build_retrieval_plan("火星学院校园卡怎么补办", index.all_documents())
    tracker = SourceTracker()
    result = await executor.execute(plan, tracker)

    # The entity-specific need has no hit, but the background part (校园卡补办)
    # should still retrieve evidence.
    assert len(result.zero_entity_hits) == 1
    assert tracker.reliable_count >= 1
    assert any(
        s.document.yuque_id == "card" for s in tracker.sources if s.reliable
    )


@pytest.mark.asyncio
async def test_executor_partial_coverage_for_multi_subquestion(executor_setup):
    executor, index, _ = executor_setup
    plan = build_retrieval_plan(
        "开甲学院大一要学什么？校园卡在哪补办？", index.all_documents()
    )
    tracker = SourceTracker()
    result = await executor.execute(plan, tracker)

    statuses = {c.status for c in result.coverage}
    assert statuses == {"DIRECT"}
    assert len(result.coverage) == 2
