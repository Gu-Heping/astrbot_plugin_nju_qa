"""Render Markdown tables as PNG images for rich message replies."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any


def _find_font(size: int):
    """Return a Pillow ImageFont using a system CJK font if available."""
    try:
        from PIL import ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for table rendering") from exc

    candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2
    )


def _is_separator_row(row: list[str]) -> bool:
    return all(re.fullmatch(r"[\s\-:]+", cell) for cell in row)


def _parse_row(line: str) -> list[str]:
    stripped = line.strip()
    inner = stripped[1:-1]
    return [cell.strip() for cell in inner.split("|")]


def _clean_cell(text: str) -> str:
    """Remove lightweight inline Markdown from a table cell."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[图片: \2]", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.strip()


def _extract_table_blocks(text: str) -> list[tuple[int, int, list[list[str]]]]:
    """Return a list of (start, end, rows) for each Markdown table in text."""
    lines = text.splitlines()
    blocks: list[tuple[int, int, list[list[str]]]] = []
    i = 0
    while i < len(lines):
        if not _is_table_row(lines[i]):
            i += 1
            continue
        start = i
        table_lines: list[list[str]] = []
        while i < len(lines) and _is_table_row(lines[i]):
            table_lines.append(_parse_row(lines[i]))
            i += 1
        end = i
        # Require at least header + separator + one row.
        if len(table_lines) >= 3:
            sep_index = next(
                (idx for idx, row in enumerate(table_lines) if _is_separator_row(row)),
                None,
            )
            if sep_index is not None and sep_index > 0:
                # Keep header and body rows, drop separator.
                rows = [table_lines[0]] + table_lines[sep_index + 1 :]
                # Ensure all rows have the same column count.
                col_count = max(len(row) for row in rows)
                normalized = [row + [""] * (col_count - len(row)) for row in rows]
                blocks.append((start, end, normalized))
        i = end
    return blocks


def _wrap_text(text: str, font: Any, max_width: int) -> list[str]:
    """Wrap text into lines that fit within max_width pixels."""
    if max_width <= 0:
        return [text]
    lines: list[str] = []
    current = ""
    for char in text:
        test = current + char
        if font.getlength(test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines or [""]


def _render_table_image(
    rows: list[list[str]],
    output_path: Path,
    font_size: int = 20,
    max_col_width: int = 360,
    padding: int = 10,
    line_height: int = 26,
    header_bg: str = "#E8E8E8",
    border_color: str = "#333333",
) -> Path:
    """Draw rows as a PNG table and save to output_path."""
    from PIL import Image, ImageDraw

    font = _find_font(font_size)
    # Calculate natural column widths (header + body).
    col_count = len(rows[0]) if rows else 0
    col_widths = [0] * col_count
    wrapped_cells: list[list[list[str]]] = []
    for r_idx, row in enumerate(rows):
        wrapped_row: list[list[str]] = []
        for c_idx, cell in enumerate(row):
            cleaned = _clean_cell(cell)
            if r_idx == 0:
                # Header: single line if it fits, otherwise wrap.
                if font.getlength(cleaned) <= max_col_width:
                    wrapped = [cleaned]
                else:
                    wrapped = _wrap_text(cleaned, font, max_col_width)
            else:
                wrapped = _wrap_text(cleaned, font, max_col_width)
            wrapped_row.append(wrapped)
            line_width = max((font.getlength(line) for line in wrapped), default=0)
            col_widths[c_idx] = max(col_widths[c_idx], int(line_width))
        wrapped_cells.append(wrapped_row)

    # Apply a minimum width and padding.
    min_col_width = font_size * 4
    col_widths = [max(w, min_col_width) + padding * 2 for w in col_widths]
    row_heights = [
        max(len(cell) * line_height + padding * 2 for cell in wrapped_row)
        for wrapped_row in wrapped_cells
    ]
    if not row_heights:
        row_heights = [line_height + padding * 2]

    table_width = sum(col_widths) + 1
    table_height = sum(row_heights) + 1
    image = Image.new("RGB", (table_width, table_height), "white")
    draw = ImageDraw.Draw(image)

    y = 0
    for r_idx, wrapped_row in enumerate(wrapped_cells):
        height = row_heights[r_idx]
        x = 0
        for c_idx, cell_lines in enumerate(wrapped_row):
            width = col_widths[c_idx]
            bg = header_bg if r_idx == 0 else "white"
            draw.rectangle([x, y, x + width, y + height], fill=bg, outline=border_color)
            text_y = y + padding
            for line in cell_lines:
                draw.text((x + padding, text_y), line, fill="black", font=font)
                text_y += line_height
            x += width
        y += height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return output_path


def render_tables_as_images(
    markdown_text: str,
    image_dir: Path,
) -> list[tuple[str, str]]:
    """Split markdown text into text/image segments.

    Returns a list of ("text", plain_text) and ("image", image_path) tuples.
    Text segments are plain Markdown with tables removed; image segments contain
    paths to rendered PNGs of the tables.
    """
    from .formatting import markdown_to_plaintext

    blocks = _extract_table_blocks(markdown_text)
    if not blocks:
        return [("text", markdown_to_plaintext(markdown_text))]

    segments: list[tuple[str, str]] = []
    last_end = 0
    for start, end, rows in blocks:
        before_lines = markdown_text.splitlines()[last_end:start]
        before_text = "\n".join(before_lines)
        if before_text.strip():
            segments.append(("text", markdown_to_plaintext(before_text)))
        image_path = image_dir / f"table_{uuid.uuid4().hex}.png"
        _render_table_image(rows, image_path)
        segments.append(("image", str(image_path)))
        last_end = end

    after_lines = markdown_text.splitlines()[last_end:]
    after_text = "\n".join(after_lines)
    if after_text.strip():
        segments.append(("text", markdown_to_plaintext(after_text)))

    return segments


def clean_table_images(image_dir: Path) -> None:
    """Remove previously rendered table images to avoid unbounded disk growth."""
    if not image_dir.exists():
        return
    for path in image_dir.glob("table_*.png"):
        try:
            path.unlink()
        except OSError:
            pass
