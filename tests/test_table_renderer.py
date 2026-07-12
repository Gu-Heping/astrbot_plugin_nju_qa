"""Tests for Markdown table rendering."""

from __future__ import annotations

from pathlib import Path

from nju_qa.table_renderer import (
    _extract_table_blocks,
    _is_separator_row,
    _parse_row,
    _render_table_image,
    render_tables_as_images,
)


def test_parse_simple_table():
    text = (
        "| 配置 | 必填 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| token | 是 | API token |\n"
    )
    blocks = _extract_table_blocks(text)
    assert len(blocks) == 1
    start, end, rows = blocks[0]
    assert start == 0
    assert end == 3
    assert rows == [
        ["配置", "必填", "说明"],
        ["token", "是", "API token"],
    ]


def test_separator_row_detection():
    assert _is_separator_row(["---", ":--:", "---"])
    assert not _is_separator_row(["配置", "必填"])


def test_parse_row_strips_padding():
    assert _parse_row("|  a  | b |") == ["a", "b"]


def test_extract_ignores_invalid_tables():
    text = "| a | b |\n| c | d |\n普通段落"
    assert _extract_table_blocks(text) == []


def test_render_table_image_creates_png(tmp_path):
    rows = [["配置", "说明"], ["token", "API token"]]
    output = tmp_path / "test.png"
    _render_table_image(rows, output)
    assert output.exists()
    from PIL import Image

    with Image.open(output) as img:
        assert img.format == "PNG"
        assert img.width > 0 and img.height > 0


def test_render_tables_as_images_splits_segments(tmp_path):
    text = (
        "请看下表：\n"
        "\n"
        "| 名称 | 值 |\n"
        "| --- | --- |\n"
        "| A | 1 |\n"
        "\n"
        "结束。"
    )
    segments = render_tables_as_images(text, tmp_path)
    assert len(segments) == 3
    assert segments[0][0] == "text"
    assert "请看下表" in segments[0][1]
    assert segments[1][0] == "image"
    assert Path(segments[1][1]).exists()
    assert segments[2][0] == "text"
    assert "结束" in segments[2][1]


def test_render_tables_as_images_without_tables(tmp_path):
    text = "这是一段没有表格的文本。"
    segments = render_tables_as_images(text, tmp_path)
    assert len(segments) == 1
    assert segments[0] == ("text", "这是一段没有表格的文本。")


def test_render_tables_as_images_handles_multiple_tables(tmp_path):
    text = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        "中间文字\n"
        "| C | D |\n| --- | --- |\n| 3 | 4 |\n"
    )
    segments = render_tables_as_images(text, tmp_path)
    images = [s for s in segments if s[0] == "image"]
    assert len(images) == 2
    for _, path in images:
        assert Path(path).exists()
