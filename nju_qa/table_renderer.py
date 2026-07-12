"""Render Markdown tables as PNG images for rich message replies."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Noto Sans CJK Simplified Chinese Regular is licensed under the SIL Open Font
# License 1.1. These URLs are tried in order until one succeeds.
_CJK_FONT_URLS = [
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://gitee.com/mirrors/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
]
_FONT_FILE_NAME = "NotoSansCJKsc-Regular.otf"
_FONT_LOCK = asyncio.Lock()


def _is_valid_font(path: str | Path) -> bool:
    """Return True if Pillow can load the given font file."""
    try:
        from PIL import ImageFont

        ImageFont.truetype(str(path), 12)
        return True
    except Exception:
        return False


def _find_font_path(font_path: str | None = None) -> str | None:
    """Return the path to a usable CJK font file, or ``None``.

    If ``font_path`` is provided and points to a valid file, it is returned.
    Otherwise the function searches common system font locations and, if
    Matplotlib is installed, scans the system font list for CJK fonts.
    """
    if font_path:
        candidate = Path(font_path)
        if candidate.is_file() and _is_valid_font(candidate):
            return str(candidate)

    candidates = [
        # Windows common Chinese fonts
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/nsimsun.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        # Linux: Noto CJK
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-VF.ttf.ttc",
        # Linux: WenQuanYi
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        # Linux: Arphic
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/opentype/arphic/uming.ttc",
        # Linux: Droid
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).is_file() and _is_valid_font(path):
            return path

    # Optional: let Matplotlib scan the system font list for CJK fonts.
    try:
        from matplotlib import font_manager as fm

        cjk_keywords = (
            "cjk",
            "hei",
            "song",
            "ming",
            "noto",
            "wqy",
            "source han",
            "han sans",
            "microsoft yahei",
            "pingfang",
            "heiti",
            "黑体",
            "宋体",
            "明体",
            "思源",
        )
        for path in fm.findSystemFonts():
            try:
                prop = fm.FontProperties(fname=path)
                name = (prop.get_name() or "").lower()
                if any(k in name for k in cjk_keywords) and _is_valid_font(path):
                    return path
            except Exception:
                continue
    except Exception:
        pass

    return None


def _find_font(size: int, font_path: str | None = None):
    """Return a Pillow ImageFont using a system CJK font if available.

    Returns ``None`` when no usable CJK font is found, so callers can fall
    back to plain text instead of producing tofu.
    """
    try:
        from PIL import ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for table rendering") from exc

    path = _find_font_path(font_path)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return None


def _has_cjk(text: str) -> bool:
    """Return True if text contains CJK characters."""
    return any(
        "一" <= char <= "鿿"
        or "　" <= char <= "〿"
        or "＀" <= char <= "￯"
        for char in text
    )


async def _download_font(url: str, dest: Path, timeout: float = 120.0) -> bool:
    """Download a font file to ``dest`` and return True if it is valid."""
    try:
        import httpx
    except ImportError:  # pragma: no cover
        logger.warning("httpx is required to download fonts automatically")
        return False

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        return _is_valid_font(dest)
    except Exception as exc:
        logger.warning("Failed to download font from %s: %s", url, exc)
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return False


async def ensure_cjk_font(
    data_dir: Path,
    configured_font_path: str | None = None,
    allow_download: bool = True,
) -> str | None:
    """Return a usable CJK font path, downloading one if necessary and allowed.

    Resolution order:
    1. ``configured_font_path`` if it is a valid font file.
    2. Any CJK font found on the system.
    3. A downloaded Noto Sans CJK SC font saved under ``data_dir/fonts``.
    4. ``None`` (callers should fall back to plain text).
    """
    if configured_font_path:
        candidate = Path(configured_font_path)
        if candidate.is_file() and _is_valid_font(candidate):
            return str(candidate)
        logger.warning("Configured font not found or invalid: %s", configured_font_path)

    system_path = _find_font_path()
    if system_path:
        return system_path

    if not allow_download:
        return None

    font_dir = data_dir / "fonts"
    dest = font_dir / _FONT_FILE_NAME
    if dest.is_file() and _is_valid_font(dest):
        return str(dest)

    async with _FONT_LOCK:
        # Double-check after acquiring the lock.
        if dest.is_file() and _is_valid_font(dest):
            return str(dest)
        for url in _CJK_FONT_URLS:
            logger.info("Downloading CJK font from %s ...", url)
            if await _download_font(url, dest):
                logger.info("CJK font saved to %s", dest)
                return str(dest)

    logger.warning(
        "Could not download a CJK font; table images will fall back to plain text"
    )
    return None


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
    font_path: str | None = None,
    font_size: int = 20,
    max_col_width: int = 360,
    padding: int = 10,
    line_height: int = 26,
    header_bg: str = "#E8E8E8",
    border_color: str = "#333333",
) -> Path:
    """Draw rows as a PNG table and save to output_path."""
    from PIL import Image, ImageDraw, ImageFont

    all_text = "".join("".join(row) for row in rows)
    font = _find_font(font_size, font_path=font_path)
    if font is None and _has_cjk(all_text):
        raise RuntimeError(
            "No CJK font found for rendering table; install a Chinese font or set table_font_path"
        )
    if font is None:
        font = ImageFont.load_default()

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
    font_path: str | None = None,
) -> list[tuple[str, str]]:
    """Split markdown text into text/image segments.

    Returns a list of ("text", plain_text) and ("image", image_path) tuples.
    Text segments are plain Markdown with tables removed; image segments contain
    paths to rendered PNGs of the tables.

    If no CJK font is available, tables are kept as plain text instead of
    producing garbled images.
    """
    from .formatting import markdown_to_plaintext

    blocks = _extract_table_blocks(markdown_text)
    if not blocks:
        return [("text", markdown_to_plaintext(markdown_text))]

    segments: list[tuple[str, str]] = []
    last_end = 0
    try:
        for start, end, rows in blocks:
            before_lines = markdown_text.splitlines()[last_end:start]
            before_text = "\n".join(before_lines)
            if before_text.strip():
                segments.append(("text", markdown_to_plaintext(before_text)))
            image_path = image_dir / f"table_{uuid.uuid4().hex}.png"
            _render_table_image(rows, image_path, font_path=font_path)
            segments.append(("image", str(image_path)))
            last_end = end
    except RuntimeError:
        # No usable font: fall back to the original plain-text rendering.
        return [("text", markdown_to_plaintext(markdown_text))]

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
