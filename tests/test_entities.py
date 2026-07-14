"""Tests for entity extraction and evidence-mode classification."""

from __future__ import annotations

import pytest

from nju_qa.entities import (
    EntityResolutionStatus,
    EntityType,
    classify_evidence_mode,
    extract_entities,
    resolve_entities,
)


def _row(path: str, title: str) -> dict:
    return {"path": path, "title": title}


def test_extract_college_major_and_campus():
    mentions = extract_entities("人工智能学院计算机科学与技术专业在仙林校区吗")
    texts = {m.text for m in mentions}
    types = {m.entity_type for m in mentions}
    assert "人工智能学院" in texts
    assert "计算机科学与技术专业" in texts
    assert "仙林校区" in texts
    assert EntityType.COLLEGE in types
    assert EntityType.MAJOR in types
    assert EntityType.CAMPUS in types


def test_extract_experimental_and_trial_class():
    mentions = extract_entities("计算机科学与技术实验班和人工智能试验班学什么")
    texts = {m.text for m in mentions}
    assert "计算机科学与技术实验班" in texts
    assert "人工智能试验班" in texts


def test_extract_year_grade_and_department():
    mentions = extract_entities("2023级中文系学生属于哪个大类")
    texts = {m.text for m in mentions}
    assert "2023级" in texts
    assert "中文系" in texts
    assert "哪个大类" not in texts  # 哪个 is an ignored prefix


def test_extract_unknown_entity_kept():
    mentions = extract_entities("星际学院在哪个校区")
    assert any(m.text == "星际学院" for m in mentions)
    # Generic interrogative prefixes are not treated as entities.
    assert not any(m.text == "哪个校区" for m in mentions)


def test_resolve_entity_matched():
    mentions = extract_entities("开甲学院大一课程")
    rows = [
        _row("QA/02_教务与学业/培养方案/开甲学院.md", "开甲学院培养方案"),
    ]
    resolved = resolve_entities(mentions, rows)
    kaijia = next(m for m in resolved if m.text == "开甲学院")
    assert kaijia.resolution_status == EntityResolutionStatus.MATCHED


def test_resolve_entity_not_found():
    mentions = extract_entities("火星学院在哪")
    rows = [_row("QA/02_教务与学业/培养方案/开甲学院.md", "开甲学院培养方案")]
    resolved = resolve_entities(mentions, rows)
    mars = next(m for m in resolved if m.text == "火星学院")
    assert mars.resolution_status == EntityResolutionStatus.NOT_FOUND


def test_resolve_entity_ambiguous_across_categories():
    mentions = extract_entities("计算机系")
    rows = [
        _row("QA/院系/计算机系.md", "计算机系"),
        _row("QA/专业/计算机系.md", "计算机系"),
    ]
    resolved = resolve_entities(mentions, rows)
    comp = next(m for m in resolved if m.text == "计算机系")
    assert comp.resolution_status == EntityResolutionStatus.AMBIGUOUS


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("你好", "NON_FACTUAL"),
        ("谢谢", "NON_FACTUAL"),
        ("你能做什么", "NON_FACTUAL"),
        ("校园卡怎么补办", "CAMPUS_FACTUAL"),
        ("开甲学院课程", "CAMPUS_FACTUAL"),
        ("宿舍床尺寸", "CAMPUS_FACTUAL"),
        ("明天会下雨吗", "NON_FACTUAL"),
    ],
)
def test_classify_evidence_mode(prompt, expected):
    assert classify_evidence_mode(prompt).value == expected
