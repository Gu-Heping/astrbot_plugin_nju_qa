import asyncio
import ast
from pathlib import Path
import httpx
import pytest
from nju_qa.answer_service import AnswerService, NO_ANSWER
from nju_qa.config import PluginConfig
from nju_qa.document_index import DocumentIndex
from nju_qa.document_store import DocumentStore
from nju_qa.models import Document
from nju_qa.retriever import HybridRetriever
from nju_qa.sync_service import SyncService
from nju_qa.yuque_client import YuqueClient


def test_main_uses_package_relative_imports():
    module = ast.parse(Path("main.py").read_text(encoding="utf-8"))
    imported = [node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
    package_imports = [
        node for node in imported if (node.module or "").startswith("nju_qa")
    ]
    assert package_imports and all(node.level == 1 for node in package_imports)


def doc(path: Path, ident="1", title="通知", body="南京大学考试安排"):
    return Document(
        ident,
        title,
        "指南",
        "nju/guide",
        title.lower(),
        f"https://www.yuque.com/nju/guide/{title.lower()}",
        "2026-01-01",
        "2026-02-01",
        body,
        path,
    )


def test_safe_paths_markdown_and_duplicates(tmp_path):
    store = DocumentStore(tmp_path)
    used = set()
    one = store.path_for("nju/guide", ["../bad"], "a/b", "1", used)
    used.add(one)
    two = store.path_for("nju/guide", ["../bad"], "a/b", "2", used)
    assert (
        one != two
        and tmp_path.resolve() in one.resolve().parents
        and ".." not in str(one)
    )
    store.write(doc(one))
    assert store.read(one).yuque_id == "1"
    with pytest.raises(ValueError):
        store.remove(Path("C:/outside.md"))


def test_config_validation():
    assert (
        PluginConfig.from_mapping({"yuque_repositories": ["nju/guide"]})
        .repositories[0]
        .namespace
        == "nju/guide"
    )
    with pytest.raises(ValueError):
        PluginConfig.from_mapping({"yuque_repositories": ["../bad"]})
    with pytest.raises(ValueError):
        PluginConfig.from_mapping({"retrieval_top_k": 0})


def test_index_keyword_vector_and_answer_sources(tmp_path):
    index = DocumentIndex(tmp_path / "a.sqlite3")
    index.open()
    d = doc(tmp_path / "a.md")
    index.upsert(d, [1.0, 0.0])
    from nju_qa.chunk_store import ChunkStore
    from nju_qa.chunking import split_markdown

    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    chunk_store.open()
    chunks = split_markdown(
        d.yuque_id,
        d.body,
        title=d.title,
        repository=d.repository,
        namespace=d.namespace,
        slug=d.slug,
        file_path=str(d.path),
        source_url=d.url,
        updated_at=d.updated_at,
    )
    chunk_store.save_document_chunks(d.yuque_id, chunks)
    config = PluginConfig.from_mapping(
        {"yuque_repositories": ["nju/guide"], "score_threshold": 0}
    )
    service = AnswerService(
        HybridRetriever(index, config, chunk_store=chunk_store),
        lambda prompt, system: asyncio.sleep(0, result="安排如下"),
    )
    result = asyncio.run(service.answer("考试安排"))
    assert "《通知》" in result and d.url in result
    assert asyncio.run(service.answer("完全无关内容")) == NO_ANSWER


def test_yuque_retries_429_then_success():
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            429 if len(calls) == 1 else 200,
            json={"data": {"name": "ok"}},
            headers={"Retry-After": "0"},
        )

    client = YuqueClient(
        "secret",
        "https://example.test",
        retries=2,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert asyncio.run(client.get_repo("nju/guide"))["name"] == "ok" and len(calls) == 2
    asyncio.run(client.close())


class FakeYuque:
    title = "通知"
    include_document = True

    async def get_repo(self, namespace):
        return {"name": "指南"}

    async def get_toc(self, namespace):
        if not self.include_document:
            return []
        return [
            {"uuid": "folder", "type": "TITLE", "title": "目录"},
            {
                "uuid": "doc",
                "parent_uuid": "folder",
                "type": "DOC",
                "id": 1,
                "url": "notice",
                "title": self.title,
            },
        ]

    async def get_document(self, namespace, slug):
        return {
            "id": 1,
            "title": self.title,
            "slug": slug,
            "body": "考试安排",
            "created_at": "a",
            "updated_at": "b",
        }


def test_sync_rename_move_delete_and_lock(tmp_path):
    config = PluginConfig.from_mapping(
        {"yuque_token": "x", "yuque_repositories": ["nju/guide"]}
    )
    store = DocumentStore(tmp_path / "docs")
    index = DocumentIndex(tmp_path / "i.sqlite3")
    from nju_qa.chunk_store import ChunkStore
    from nju_qa.vector_index import ChunkVectorIndex

    chunk_store = ChunkStore(tmp_path / "chunks.sqlite3")
    vector_index = ChunkVectorIndex(
        tmp_path / "vectors", model="mock", embedding_dimension=64
    )
    chunk_store.open()
    api = FakeYuque()

    async def embed(text):
        return [0.1] * 64

    sync = SyncService(
        config,
        api,
        store,
        index,
        chunk_store=chunk_store,
        vector_index=vector_index,
        embed=embed,
    )
    result = asyncio.run(sync.sync_all())
    assert result.succeeded == 1 and len(index.all_documents()) == 1
    assert chunk_store.chunk_count() > 0
    api.title = "重命名通知"
    result = asyncio.run(sync.sync_all())
    assert result.succeeded == 1 and len(list((tmp_path / "docs").rglob("*.md"))) == 1
    assert chunk_store.get_document_chunks("1")[0].title == "重命名通知"
    api.include_document = False
    result = asyncio.run(sync.sync_all())
    assert result.deleted == 1 and not index.all_documents()
    assert chunk_store.chunk_count() == 0
    assert vector_index.count() == 0
