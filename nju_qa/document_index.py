from __future__ import annotations
import json
import math
import sqlite3
from pathlib import Path
from .models import Document


class DocumentIndex:
    def __init__(self, path: Path):
        self.path, self.conn = path, None

    def open(self) -> None:
        if self.conn:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (yuque_id TEXT PRIMARY KEY,title TEXT,repository TEXT,namespace TEXT,slug TEXT,url TEXT,created_at TEXT,updated_at TEXT,body TEXT,path TEXT,embedding TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT)"
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def upsert(self, doc: Document, embedding: list[float] | None = None) -> None:
        self.open()
        self.conn.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(yuque_id) DO UPDATE SET title=excluded.title,repository=excluded.repository,namespace=excluded.namespace,slug=excluded.slug,url=excluded.url,created_at=excluded.created_at,updated_at=excluded.updated_at,body=excluded.body,path=excluded.path,embedding=COALESCE(excluded.embedding,documents.embedding)",
            (
                doc.yuque_id,
                doc.title,
                doc.repository,
                doc.namespace,
                doc.slug,
                doc.url,
                doc.created_at,
                doc.updated_at,
                doc.body,
                str(doc.path or ""),
                json.dumps(embedding) if embedding else None,
            ),
        )
        self.conn.commit()

    def delete_missing(self, namespace: str, ids: set[str]) -> list[sqlite3.Row]:
        self.open()
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE namespace=?", (namespace,)
        ).fetchall()
        doomed = [r for r in rows if r["yuque_id"] not in ids]
        self.conn.executemany(
            "DELETE FROM documents WHERE yuque_id=?", [(r["yuque_id"],) for r in doomed]
        )
        self.conn.commit()
        return doomed

    def all_documents(self):
        self.open()
        return self.conn.execute("SELECT * FROM documents").fetchall()

    def keyword(self, query: str, limit: int):
        self.open()
        terms = [x for x in query.casefold().split() if x]
        rows = self.all_documents()
        return [
            (r, sum((r["title"] + " " + r["body"]).casefold().count(t) for t in terms))
            for r in rows
            if any(t in (r["title"] + r["body"]).casefold() for t in terms)
        ][:limit]

    def vector(self, vector: list[float], limit: int):
        scored = []
        for row in self.all_documents():
            if not row["embedding"]:
                continue
            other = json.loads(row["embedding"])
            den = math.sqrt(sum(x * x for x in vector)) * math.sqrt(
                sum(x * x for x in other)
            )
            scored.append(
                (row, sum(a * b for a, b in zip(vector, other)) / den if den else 0)
            )
        return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]

    def set_state(self, key: str, value: str):
        self.open()
        self.conn.execute(
            "INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_state(self, key: str):
        self.open()
        row = self.conn.execute(
            "SELECT value FROM state WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
