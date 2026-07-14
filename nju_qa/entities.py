"""Entity mention extraction and evidence-mode classification for NJU queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueryEvidenceMode(str, Enum):
    """High-level evidence requirement for a user prompt."""

    NON_FACTUAL = "NON_FACTUAL"
    CAMPUS_FACTUAL = "CAMPUS_FACTUAL"


class EntityType(str, Enum):
    """Kinds of campus entities that anchor entity-specific retrieval."""

    COLLEGE = "学院"
    ACADEMY = "书院"
    DEPARTMENT = "系"
    MAJOR = "专业"
    EXPERIMENTAL_CLASS = "实验班"
    TRIAL_CLASS = "试验班"
    BROAD_CATEGORY = "大类"
    CAMPUS = "校区"
    CENTER = "中心"
    DEPARTMENT_ORG = "部门"
    YEAR_GRADE = "年级"


class EntityResolutionStatus(str, Enum):
    """Whether an extracted entity could be located in the synced metadata."""

    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True)
class EntityMention:
    """One entity mention extracted from the user query."""

    text: str
    entity_type: EntityType
    resolution_status: EntityResolutionStatus = EntityResolutionStatus.NOT_FOUND
    matched_paths: tuple[str, ...] = ()

    @property
    def is_known(self) -> bool:
        return self.resolution_status != EntityResolutionStatus.NOT_FOUND


# Type suffix with a preceding name.  Non-greedy so the shortest sensible name is
# captured.  English letters and digits are allowed for labs/centres such as
# "AI学院" or "IIIA中心".
_TYPE_PATTERNS: list[tuple[EntityType, re.Pattern]] = [
    (EntityType.COLLEGE, re.compile(r"([一-龥A-Za-z0-9]+?)学院")),
    (EntityType.ACADEMY, re.compile(r"([一-龥A-Za-z0-9]+?)书院")),
    (EntityType.DEPARTMENT, re.compile(r"([一-龥A-Za-z0-9]+?)系")),
    (EntityType.MAJOR, re.compile(r"([一-龥A-Za-z0-9]+?)专业")),
    (EntityType.EXPERIMENTAL_CLASS, re.compile(r"([一-龥A-Za-z0-9]+?)实验班")),
    (EntityType.TRIAL_CLASS, re.compile(r"([一-龥A-Za-z0-9]+?)试验班")),
    (EntityType.BROAD_CATEGORY, re.compile(r"([一-龥A-Za-z0-9]+?)大类")),
    (EntityType.CAMPUS, re.compile(r"([一-龥A-Za-z0-9]+?)校区")),
    (EntityType.CENTER, re.compile(r"([一-龥A-Za-z0-9]+?)中心")),
    (EntityType.DEPARTMENT_ORG, re.compile(r"([一-龥A-Za-z0-9]+?)部门")),
    (EntityType.YEAR_GRADE, re.compile(r"(20\d{2})级")),
]

# Prefixes that should not be treated as part of an entity name.
_IGNORE_PREFIXES: frozenset[str] = frozenset(
    "什么 哪个 哪 某 其他 别的 相关 对应 该 这个 那个 我们 你们 他们 的 之 和 与 或 在 属于哪个".split()
)

# Generic campus phrases that are not specific entities even if the regex matches.
_GENERIC_WORDS: frozenset[str] = frozenset(
    "南京大学 南大 校园 学生 新生 老生 本科生 研究生 教务 课程 学分 考试 成绩 "
    "培养方案 选课 退课 补考 重修 图书馆 宿舍 食堂 交通 奖助 奖学金 助学金 "
    "入学 报到 毕业 就业 户口 档案 党组织 团组织 军训 体检 缴费 学费 住宿 "
    "校园卡 学生证 身份证 院系 学院部门".split()
)


def _clean_entity_prefix(text: str) -> str | None:
    """Remove generic/ignored prefixes and validate the remaining name."""
    text = text.strip()
    if not text:
        return None

    # Strip leading generic campus words.
    for generic in sorted(_GENERIC_WORDS, key=len, reverse=True):
        if text.startswith(generic):
            text = text[len(generic) :].strip()
    if not text:
        return None

    # Strip leading interrogative/generic prefixes.
    for prefix in sorted(_IGNORE_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if not text:
        return None

    # Drop purely grammatical residue.
    if re.fullmatch(r"[的之乎者也是了在在和有吗呢吧？?，。；：\s]+", text):
        return None

    return text


def extract_entities(query: str) -> list[EntityMention]:
    """Extract campus entity mentions from ``query`` without KB lookup.

    Unknown entities are returned with ``resolution_status=NOT_FOUND`` so the
    retrieval plan can still search for them explicitly.  Overlapping matches
    are resolved by accepting shorter/earlier entities first and trimming later
    matches so that names do not swallow preceding entities.
    """
    raw_matches: list[tuple[int, int, EntityType, str, str]] = []
    for entity_type, pattern in _TYPE_PATTERNS:
        for match in pattern.finditer(query):
            if entity_type is EntityType.YEAR_GRADE:
                # Keep the original surface form, e.g. "2023级".
                prefix = match.group(1)
                cleaned = _clean_entity_prefix(prefix)
                if cleaned is None or len(cleaned) != 4:
                    continue
                full_text = match.group(0)
                start = match.start(0)
                end = match.end(0)
            else:
                prefix = match.group(1)
                cleaned = _clean_entity_prefix(prefix)
                if cleaned is None or len(cleaned) < 2:
                    continue
                full_text = cleaned + entity_type.value
                if full_text in _GENERIC_WORDS:
                    continue
                # start/end are character offsets in the original query.
                start = match.start(1)
                end = match.end()
            raw_matches.append((start, end, entity_type, full_text, cleaned))

    if not raw_matches:
        return []

    # Prefer shorter matches at the same start so nested entities are split
    # (e.g. "人工智能学院计算机科学与技术专业" → two entities, not one long major).
    raw_matches.sort(key=lambda x: (x[0], x[1] - x[0]))

    mentions: list[EntityMention] = []
    last_end = -1
    for start, end, entity_type, full_text, prefix in raw_matches:
        if start < last_end:
            # Overlap with an already-accepted entity: trim the new match to
            # begin right after the previous one and re-validate the prefix.
            suffix_len = end - start - len(prefix)
            new_start = last_end
            new_prefix = query[new_start : end - suffix_len]
            cleaned = _clean_entity_prefix(new_prefix)
            if cleaned is None:
                continue
            if entity_type is EntityType.YEAR_GRADE:
                if len(cleaned) != 4:
                    continue
                full_text = query[new_start:end]
            else:
                if len(cleaned) < 2:
                    continue
                full_text = cleaned + entity_type.value
                if full_text in _GENERIC_WORDS:
                    continue
            start = new_start
            prefix = cleaned

        mentions.append(EntityMention(text=full_text, entity_type=entity_type))
        last_end = end

    return mentions


def _row_path(row: Any) -> str:
    if hasattr(row, "get"):
        return str(row.get("path") or "")
    return str(row["path"] or "")


def _row_title(row: Any) -> str:
    if hasattr(row, "get"):
        return str(row.get("title") or "")
    return str(row["title"] or "")


def resolve_entities(
    mentions: list[EntityMention], rows: list[Any]
) -> list[EntityMention]:
    """Match extracted entities against document paths/titles."""
    if not rows:
        return mentions

    resolved: list[EntityMention] = []
    for mention in mentions:
        matches: list[str] = []
        for row in rows:
            path = _row_path(row)
            title = _row_title(row)
            if mention.text in path or mention.text in title:
                matches.append(path or title)
        if not matches:
            status = EntityResolutionStatus.NOT_FOUND
        elif len(matches) == 1:
            status = EntityResolutionStatus.MATCHED
        else:
            namespaces: set[str] = set()
            categories: set[str] = set()
            for m in matches:
                segs = [s for s in m.replace("\\", "/").split("/") if s]
                if segs:
                    namespaces.add(segs[0])
                if len(segs) > 1:
                    categories.add(segs[1])
            status = (
                EntityResolutionStatus.AMBIGUOUS
                if len(namespaces) > 1 or len(categories) > 1
                else EntityResolutionStatus.MATCHED
            )
        resolved.append(
            EntityMention(
                text=mention.text,
                entity_type=mention.entity_type,
                resolution_status=status,
                matched_paths=tuple(matches[:5]),
            )
        )
    return resolved


# Patterns that are clearly non-factual small talk.
_NON_FACTUAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(你好|您好|嗨|哈喽|hello|hi|hey|在吗|在不在|早上好|下午好|晚上好)([！!。，,\s]|$)", re.IGNORECASE),
    re.compile(r"^(谢谢|多谢|感谢|拜拜|再见|bye|goodbye)([！!。，,\s]|$)", re.IGNORECASE),
    re.compile(r"^(你?是谁|你叫什么|你叫什么名字|介绍一下你|自我介绍一下)"),
    re.compile(r"^(你?能做什么|你?会什么|你?有什么功能|帮助|help|怎么用)"),
    re.compile(r"^(好的|行|可以|ok|okay|知道了|明白)([！!。，,\s]|$)", re.IGNORECASE),
]

_CAMPUS_MARKERS: tuple[str, ...] = (
    "南京大学", "南大", "校园", "校区", "学院", "书院", "专业", "系", "大类",
    "实验班", "试验班", "教务", "课程", "学分", "考试", "宿舍", "食堂", "校园卡",
    "学生证", "奖助", "军训", "报到", "入学", "毕业", "就业", "户口", "档案",
)

_FACTUAL_HINTS: tuple[str, ...] = (
    "哪里", "在哪", "地点", "位置", "怎么", "如何", "多少", "费用", "多少钱",
    "时间", "何时", "什么时候", "流程", "要求", "条件", "办理", "补办", "网址",
    "网站", "入口", "电话", "联系方式", "截止日期", "期限",
)


def classify_evidence_mode(prompt: str) -> QueryEvidenceMode:
    """Return ``NON_FACTUAL`` only for pure small talk; everything else is factual."""
    if not prompt or not prompt.strip():
        return QueryEvidenceMode.NON_FACTUAL

    text = prompt.strip().casefold()
    for pattern in _NON_FACTUAL_PATTERNS:
        if pattern.search(text):
            return QueryEvidenceMode.NON_FACTUAL

    if any(marker in prompt for marker in _CAMPUS_MARKERS):
        return QueryEvidenceMode.CAMPUS_FACTUAL

    if any(hint in prompt for hint in _FACTUAL_HINTS):
        return QueryEvidenceMode.CAMPUS_FACTUAL

    return QueryEvidenceMode.NON_FACTUAL
