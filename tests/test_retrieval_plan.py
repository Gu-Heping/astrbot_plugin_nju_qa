"""Tests for retrieval planning and evidence coverage classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from nju_qa.models import Document, SearchResult
from nju_qa.retrieval_plan import (
    CoverageStatus,
    build_retrieval_plan,
    check_coverage,
    classify_coverage,
)


def _doc_row(path: str, title: str, yuque_id: str, repository: str = "repo") -> dict:
    return {
        "path": path,
        "title": title,
        "yuque_id": yuque_id,
        "repository": repository,
        "namespace": path.split("/")[0],
        "slug": Path(path).stem,
        "url": f"https://yuque.test/{path.split('/')[0]}/{Path(path).stem}",
        "created_at": "a",
        "updated_at": "b",
        "body": "",
    }


@pytest.fixture
def rows():
    return [
        _doc_row("QA/01_入学与行政事务/校园卡.md", "校园卡", "card"),
        _doc_row("QA/02_教务与学业/培养方案/开甲学院.md", "开甲学院培养方案", "kaijia"),
        _doc_row("QA/02_教务与学业/选课.md", "选课", "course"),
        _doc_row("Other/通用.md", "通用校园介绍", "general", repository="other"),
    ]


def test_plan_extracts_entity_terms(rows):
    plan = build_retrieval_plan("开甲学院大一要学什么", rows)
    assert len(plan) == 1
    assert "开甲学院" in plan[0].entity_terms
    assert plan[0].scope_namespace == "QA"


def test_plan_splits_subquestions(rows):
    plan = build_retrieval_plan("校园卡在哪补办？选课系统怎么进？", rows)
    assert len(plan) == 2
    questions = {p.question for p in plan}
    assert any("校园卡" in q for q in questions)
    assert any("选课" in q for q in questions)


def test_classify_direct_requires_entity_in_source(rows):
    plan = build_retrieval_plan("开甲学院大一要学什么", rows)[0]
    direct_source = SearchResult(
        source_id="S1",
        document=Document(
            yuque_id="kaijia",
            title="开甲学院培养方案",
            repository="repo",
            namespace="QA",
            slug="kaijia",
            url="",
            created_at="a",
            updated_at="b",
            body="开甲学院大一学习数学、程序设计。",
            path=Path("QA/02_教务与学业/培养方案/开甲学院.md"),
        ),
        score=1.0,
        reliable=True,
    )
    generic_source = SearchResult(
        source_id="S2",
        document=Document(
            yuque_id="general",
            title="通用校园介绍",
            repository="other",
            namespace="Other",
            slug="general",
            url="",
            created_at="a",
            updated_at="b",
            body="大一新生都要学习通识课和体育课。",
            path=Path("Other/通用.md"),
        ),
        score=0.5,
        reliable=True,
    )

    assert classify_coverage(plan, [direct_source]).status == CoverageStatus.DIRECT
    assert classify_coverage(plan, [generic_source]).status == CoverageStatus.BACKGROUND
    assert classify_coverage(plan, []).status == CoverageStatus.UNSUPPORTED


def test_check_coverage_flags_entity_specific_unsupported(rows):
    plan = build_retrieval_plan("开甲学院大一要学什么", rows)
    generic_source = SearchResult(
        source_id="S2",
        document=Document(
            yuque_id="general",
            title="通用校园介绍",
            repository="other",
            namespace="Other",
            slug="general",
            url="",
            created_at="a",
            updated_at="b",
            body="大一新生都要学习通识课。",
            path=Path("Other/通用.md"),
        ),
        score=0.5,
        reliable=True,
    )
    coverage = check_coverage(plan, [generic_source])
    assert coverage[0].status == CoverageStatus.BACKGROUND


def test_generic_question_direct_without_entity(rows):
    plan = build_retrieval_plan("校园卡怎么补办", rows)[0]
    source = SearchResult(
        source_id="S1",
        document=Document(
            yuque_id="card",
            title="校园卡",
            repository="repo",
            namespace="QA",
            slug="card",
            url="",
            created_at="a",
            updated_at="b",
            body="校园卡可在信息化建设管理服务中心补办。",
            path=Path("QA/01_入学与行政事务/校园卡.md"),
        ),
        score=1.0,
        reliable=True,
    )
    assert classify_coverage(plan, [source]).status == CoverageStatus.DIRECT
