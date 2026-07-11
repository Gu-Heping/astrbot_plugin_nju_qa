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


def read_document_content(
    root: Path,
    file_path: str,
    offset: int = 0,
    limit: int = 12000,
    strip_metadata: bool = True,
) -> dict:
    if offset < 0 or not 1 <= limit <= 20000:
        raise ValueError("invalid pagination")
    candidate = (root / file_path).resolve()
    if (
        ".." in Path(file_path).parts
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise ValueError("invalid document path")
    candidate.relative_to(root.resolve())
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
