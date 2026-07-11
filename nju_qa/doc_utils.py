"""Shared, safe document contract used by every knowledge-base tool."""

from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
import yaml


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    return (yaml.safe_load(parts[1]) or {}, parts[2] if len(parts) == 3 else "")


def strip_meta_table(body: str) -> str:
    return re.sub(
        r"\A\s*\|[^\n]*\|\n\|(?:[-: ]+\|)+\n(?:\|[^\n]*\|\n)?", "", body
    ).lstrip()


def clean_document_body(body: str) -> str:
    """Return a display-friendly version of a Markdown document body.

    Removes YAML frontmatter, leading metadata tables, markdown images and HTML
    tags, while preserving link text and URLs.
    """
    body = strip_meta_table(parse_frontmatter(body)[1])
    # Drop markdown images entirely.
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    # Keep both link text and URL.
    body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", body)
    # Drop HTML tags (including Yuque font/color spans).
    body = re.sub(r"<[^>]+>", "", body)
    # Normalize whitespace.
    body = re.sub(r"\s+", " ", body).strip()
    return body


def read_document_content(
    root: Path,
    file_path: str,
    offset: int = 0,
    limit: int = 12000,
    strip_metadata: bool = True,
) -> dict:
    if offset < 0 or not 1 <= limit <= 20000:
        raise ValueError("invalid pagination")
    root_resolved = root.resolve()
    file_path_obj = Path(file_path)
    if ".." in file_path_obj.parts:
        raise ValueError("invalid document path")
    # file_path may already include the root prefix (e.g. stored relative to cwd).
    candidate = (root_resolved / file_path_obj).resolve()
    if not candidate.is_relative_to(root_resolved):
        alt = file_path_obj.resolve()
        if alt.is_relative_to(root_resolved):
            candidate = alt
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not candidate.is_relative_to(root_resolved)
    ):
        raise ValueError("invalid document path")
    metadata, body = parse_frontmatter(candidate.read_text(encoding="utf-8"))
    body = strip_meta_table(body) if strip_metadata else body
    chunk = body[offset : offset + limit]
    return {
        "metadata": metadata,
        "content": chunk,
        "total_chars": len(body),
        "has_more": offset + len(chunk) < len(body),
        "next_offset": offset + len(chunk),
        "offset": offset,
    }


def parse_yuque_doc_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    parts = [unquote(x) for x in parsed.path.split("/") if x]
    return ("/".join(parts[:-1]), parts[-1]) if len(parts) >= 2 else None


def doc_record_to_public_dict(row) -> dict:
    return {
        key: row[key]
        for key in (
            "yuque_id",
            "title",
            "repository",
            "namespace",
            "slug",
            "url",
            "created_at",
            "updated_at",
            "path",
        )
    }
