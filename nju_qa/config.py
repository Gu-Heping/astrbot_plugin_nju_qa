from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
import re
from .models import Repository


@dataclass(frozen=True)
class PluginConfig:
    yuque_token: str = ""
    yuque_base_url: str = "https://www.yuque.com/api/v2"
    repositories: tuple[Repository, ...] = ()
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    enable_vector_search: bool = True
    wake_words: tuple[str, ...] = ("南大助手", "南小答", "nju")
    enable_private_chat: bool = True
    enable_group_at: bool = True
    retrieval_top_k: int = 5
    score_threshold: float = 0.25
    chunk_size: int = 1200
    chunk_overlap: int = 180
    group_rate_limit: int = 30
    group_rate_limit_window: int = 3600
    private_rate_limit: int = 20
    private_rate_limit_window: int = 3600

    @classmethod
    def from_mapping(cls, raw: Any) -> "PluginConfig":
        raw = raw or {}
        repos = raw.get("yuque_repositories", [])
        if isinstance(repos, str):
            repos = [x.strip() for x in repos.split(",") if x.strip()]
        parsed = []
        for item in repos:
            if isinstance(item, str):
                ns, name = item, ""
            elif isinstance(item, dict):
                ns, name = str(item.get("namespace", "")), str(item.get("name", ""))
            else:
                raise ValueError("yuque_repositories 必须是 namespace 字符串或对象数组")
            if not re.fullmatch(
                r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", ns
            ):
                raise ValueError("无效的知识库 namespace")
            parsed.append(Repository(ns, name))
        base = str(raw.get("yuque_base_url", cls.yuque_base_url)).rstrip("/")
        if urlparse(base).scheme not in {"http", "https"}:
            raise ValueError("yuque_base_url 必须是 HTTP(S) URL")
        words = raw.get("wake_words", cls.wake_words)
        if isinstance(words, str):
            words = tuple(x.strip() for x in words.split(",") if x.strip())
        top_k = int(raw.get("retrieval_top_k", 5))
        threshold = float(raw.get("score_threshold", 0.25))
        chunk_size = int(raw.get("chunk_size", 1200))
        chunk_overlap = int(raw.get("chunk_overlap", 180))
        group_rate_limit = int(raw.get("group_rate_limit", 30))
        group_rate_limit_window = int(raw.get("group_rate_limit_window", 3600))
        private_rate_limit = int(raw.get("private_rate_limit", 20))
        private_rate_limit_window = int(raw.get("private_rate_limit_window", 3600))
        if not 1 <= top_k <= 20 or not 0 <= threshold <= 1:
            raise ValueError("检索配置超出允许范围")
        if not 200 <= chunk_size <= 8000 or not 0 <= chunk_overlap < chunk_size // 2:
            raise ValueError("chunk 配置超出允许范围")
        if not 0 <= group_rate_limit <= 1000 or not 0 <= private_rate_limit <= 1000:
            raise ValueError("rate_limit 必须在 0 到 1000 之间")
        if not 60 <= group_rate_limit_window <= 86400 or not 60 <= private_rate_limit_window <= 86400:
            raise ValueError("rate_limit_window 必须在 60 到 86400 秒之间")
        return cls(
            str(raw.get("yuque_token", "")),
            base,
            tuple(parsed),
            str(raw.get("embedding_api_key", "")),
            str(raw.get("embedding_base_url", "")),
            str(raw.get("embedding_model", "text-embedding-3-small")),
            bool(raw.get("enable_vector_search", True)),
            tuple(words),
            bool(raw.get("enable_private_chat", True)),
            bool(raw.get("enable_group_at", True)),
            top_k,
            threshold,
            chunk_size,
            chunk_overlap,
            group_rate_limit,
            group_rate_limit_window,
            private_rate_limit,
            private_rate_limit_window,
        )
