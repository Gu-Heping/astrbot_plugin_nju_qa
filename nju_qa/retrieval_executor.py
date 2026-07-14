"""Code-level executor for the retrieval plan.

The executor runs entity-specific and background retrieval deterministically
before the LLM is asked to compose an answer.  This prevents the model from
ignoring the plan and ensures that zero-entity-hit cases are reported explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .evidence import grep_hits_to_search_results
from .models import SearchResult
from .retrieval_plan import CoverageResult, CoverageStatus, RetrievalNeed, check_coverage

if TYPE_CHECKING:
    from .agent import SourceTracker
    from .document_index import DocumentIndex
    from .retriever import HybridRetriever


@dataclass
class NeedResult:
    """Retrieval outcome for one planned need."""

    need: RetrievalNeed
    entity_hits: list[SearchResult] = field(default_factory=list)
    background_hits: list[SearchResult] = field(default_factory=list)
    zero_entity_hit: bool = False


@dataclass
class ExecutionResult:
    """Aggregated result of executing a full retrieval plan."""

    need_results: list[NeedResult] = field(default_factory=list)
    zero_entity_hits: list[RetrievalNeed] = field(default_factory=list)
    coverage: list[CoverageResult] = field(default_factory=list)

    @property
    def has_direct(self) -> bool:
        return any(c.status == CoverageStatus.DIRECT for c in self.coverage)

    @property
    def has_background(self) -> bool:
        return any(c.status == CoverageStatus.BACKGROUND for c in self.coverage)


class RetrievalExecutor:
    """Execute a :class:`RetrievalNeed` plan and record evidence in a tracker."""

    def __init__(
        self,
        retriever: HybridRetriever | None,
        index: DocumentIndex | None,
        docs_root: Path | None = None,
    ):
        self.retriever = retriever
        self.index = index
        self.docs_root = docs_root

    async def execute(
        self, plan: list[RetrievalNeed], tracker: SourceTracker
    ) -> ExecutionResult:
        """Run entity and background retrieval for every need in ``plan``."""
        need_results: list[NeedResult] = []
        for need in plan:
            entity_hits, background_hits = await self._execute_need(need, tracker)
            zero = bool(
                need.entity_terms
                and not any(hit.reliable for hit in entity_hits)
            )
            need_results.append(
                NeedResult(
                    need=need,
                    entity_hits=entity_hits,
                    background_hits=background_hits,
                    zero_entity_hit=zero,
                )
            )

        zero_entity_hits = [nr.need for nr in need_results if nr.zero_entity_hit]
        coverage = check_coverage(plan, tracker.sources, require_reliable=True)
        return ExecutionResult(
            need_results=need_results,
            zero_entity_hits=zero_entity_hits,
            coverage=coverage,
        )

    async def _execute_need(
        self, need: RetrievalNeed, tracker: SourceTracker
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        entity_hits: list[SearchResult] = []
        background_hits: list[SearchResult] = []

        if need.entity_terms:
            # Grep using the entity name itself.  Adding arbitrary core terms can
            # pull in cross-boundary bigrams (e.g. "院校") and make the grep
            # fail even when a clean entity match exists.
            entity_hits = await self._grep(
                need.entity_terms,
                namespace=need.scope_namespace,
                path_prefix=need.scope_path_prefix,
            )
            # Fallback to hybrid search when grep finds nothing reliable.
            if not any(hit.reliable for hit in entity_hits):
                fallback = await self._search(
                    " ".join(need.entity_terms),
                    namespace=need.scope_namespace,
                    path_prefix=need.scope_path_prefix,
                )
                entity_hits = self._dedup_merge(entity_hits, fallback)
            # A hit is only an entity hit when the entity text is actually present.
            entity_hits = [
                hit for hit in entity_hits if self._contains_terms(hit, need.entity_terms)
            ]
            if entity_hits:
                tracker.add(entity_hits)

        # Background retrieval uses the core/question terms without the entity so
        # generic campus background does not accidentally satisfy entity-specific
        # coverage.
        bg_terms = need.core_terms + need.question_terms[:3]
        if need.entity_terms:
            # Remove the entity text from the background query.
            entity_set = set(need.entity_terms)
            bg_terms = [t for t in bg_terms if t not in entity_set]
        if bg_terms:
            background_hits = await self._search(
                " ".join(bg_terms),
                namespace=need.scope_namespace,
                path_prefix=need.scope_path_prefix,
            )
            if background_hits:
                tracker.add(background_hits)

        return entity_hits, background_hits

    async def _grep(
        self,
        terms: list[str],
        *,
        namespace: str = "",
        path_prefix: str = "",
    ) -> list[SearchResult]:
        if not terms or self.index is None or self.docs_root is None:
            return []
        # Import lazily to avoid a circular import at module load time.
        from .tools.documents import GrepLocalDocsTool

        tool = GrepLocalDocsTool(index=self.index, docs_root=self.docs_root)
        hits = tool._search(
            terms,
            repo_filter="",
            namespace=namespace,
            path_prefix=path_prefix,
            document_ids=None,
            include_archived=False,
            context_lines=2,
            limit=10,
        )
        return grep_hits_to_search_results(hits, terms)

    async def _search(
        self,
        query: str,
        *,
        namespace: str = "",
        path_prefix: str = "",
    ) -> list[SearchResult]:
        if not query.strip() or self.retriever is None:
            return []
        scope: dict[str, Any] = {"include_archived": False}
        if namespace:
            scope["namespace"] = namespace
        if path_prefix:
            scope["path_prefix"] = path_prefix
        return await self.retriever.search(query, **scope)

    @staticmethod
    def _contains_terms(hit: SearchResult, terms: list[str]) -> bool:
        """Return True when ``hit`` contains every required ``terms`` literally."""
        if not terms:
            return True
        text = ""
        if hit.chunk is not None and hit.chunk.content_snippet:
            text = hit.chunk.content_snippet
        elif hit.document is not None:
            text = hit.document.body
        norm = text.casefold()
        return all(term.casefold() in norm for term in terms)

    @staticmethod
    def _dedup_merge(
        a: list[SearchResult], b: list[SearchResult]
    ) -> list[SearchResult]:
        by_id: dict[str, SearchResult] = {}
        for result in a + b:
            key = result.document.yuque_id or result.document.url or result.document.title
            existing = by_id.get(key)
            if existing is None or result.score > existing.score:
                by_id[key] = result
        return sorted(by_id.values(), key=lambda r: r.score, reverse=True)
