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


def test_plan_extracts_unknown_entities(rows):
    plan = build_retrieval_plan("星际学院怎么进", rows)
    assert len(plan) == 1
    assert "星际学院" in plan[0].entity_terms
    # Unknown entities still get a scoped search when possible.
    assert plan[0].scope_namespace == ""


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


def test_classify_ignores_unreliable_sources(rows):
    plan = build_retrieval_plan("开甲学院大一要学什么", rows)[0]
    unreliable = SearchResult(
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
        reliable=False,
    )
    assert classify_coverage(plan, [unreliable]).status == CoverageStatus.UNSUPPORTED


def test_classify_direct_requires_same_window(rows):
    plan = build_retrieval_plan("开甲学院大一要学什么", rows)[0]
    # Entity and core condition appear in different windows.
    source = SearchResult(
        source_id="S1",
        document=Document(
            yuque_id="split",
            title="分散文档",
            repository="repo",
            namespace="QA",
            slug="split",
            url="",
            created_at="a",
            updated_at="b",
            body="开甲学院简介。\n\n大一学生都要学习数学。",
            path=Path("QA/split.md"),
        ),
        score=1.0,
        reliable=True,
    )
    assert classify_coverage(plan, [source]).status == CoverageStatus.BACKGROUND


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


def test_no_answer_window_prevents_direct(rows):
    plan = build_retrieval_plan("校园卡过期是什么意思", rows)
    source = SearchResult(
        source_id="S1",
        document=Document(
            yuque_id="faq",
            title="常见疑问汇总",
            repository="repo",
            namespace="QA",
            slug="faq",
            url="",
            created_at="a",
            updated_at="b",
            body="校园卡过期暂无可靠答案，待补充。",
            path=Path("QA/faq.md"),
        ),
        score=1.0,
        reliable=True,
    )
    assert classify_coverage(plan, [source]).status == CoverageStatus.UNSUPPORTED
