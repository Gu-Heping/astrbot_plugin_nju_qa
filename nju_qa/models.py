from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Repository:
    namespace: str
    name: str = ""


@dataclass(frozen=True)
class Document:
    yuque_id: str
    title: str
    repository: str
    namespace: str
    slug: str
    url: str
    created_at: str
    updated_at: str
    body: str
    path: Path | None = None


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    document: Document
    score: float


@dataclass
class SyncResult:
    succeeded: int = 0
    failed: int = 0
    deleted: int = 0
    skipped: int = 0

    def summary(self) -> str:
        return f"同步完成：成功 {self.succeeded}，失败 {self.failed}，删除 {self.deleted}，跳过 {self.skipped}。"
