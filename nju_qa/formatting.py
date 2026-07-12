"""Convert Markdown-flavoured text into a plain-text style suitable for QQ."""

from __future__ import annotations

import re


def markdown_to_plaintext(text: str) -> str:
    """Remove or rewrite Markdown markup so the result looks fine in plain text.

    Headings, bold/italic/strikethrough, code fences, inline code, blockquotes,
    images and most list markers are normalised. URLs inside links are preserved.
    """
    if not text:
        return text

    # Fenced code blocks: keep content, drop fences and language hint.
    text = re.sub(
        r"```[\s\S]*?```",
        lambda m: _clean_codeblock(m.group(0)),
        text,
    )

    # Inline code -> plain text (remove backticks).
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Images -> remove or replace with a short marker.
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[图片: \2]", text)

    # Links -> text (url).
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Headings -> remove # markers.
    text = re.sub(r"^#{1,6}\s+(.*)$", r"\1", text, flags=re.MULTILINE)

    # Horizontal rules.
    text = re.sub(r"^[\*\-_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Blockquotes -> prefix with "| ".
    text = re.sub(r"^>\s?", "| ", text, flags=re.MULTILINE)

    # Bold / italic / strikethrough (must come after list-marker handling).
    # Double emphasis first.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Single emphasis; avoid list items starting with "* " by requiring the
    # closing marker not to be followed by whitespace + end of line.
    text = re.sub(r"(?m)\*([^*\n]+)\*(?!\s*$)", r"\1", text)
    text = re.sub(r"(?m)_([^_\n]+)_(?!\s*$)", r"\1", text)

    # Unordered list markers: normalise to "- ".
    text = re.sub(r"^[\*\+]\s+", "- ", text, flags=re.MULTILINE)

    # Collapse multiple blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _clean_codeblock(block: str) -> str:
    """Drop the opening/closing ``` fences and any language hint."""
    lines = block.splitlines()
    if len(lines) >= 2:
        # Opening fence may contain language, e.g. ```python
        content = lines[1:-1]
    else:
        content = lines
    return "\n".join(content)
