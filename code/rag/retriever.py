"""
Hybrid retriever — BM25 sparse search + ChromaDB dense search + cross-encoder reranking.
"""
from typing import Any, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config.settings import settings
from models.schemas import Domain, RetrievedChunk, RetrievalResult
from rag.corpus_store import CorpusStore

_embed_model: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None


def _get_embed_model() -> SentenceTransformer:
    ### use of this function: get embed model
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    ### use of this function: get cross encoder
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


class HybridRetriever:
    """Combines BM25 sparse, dense vector, and cross-encoder reranking."""

    def __init__(self, store: CorpusStore) -> None:
        ### use of this function: init
        self._store = store
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[dict[str, Any]]]] = {}

    def _build_bm25_index(self, domain: Domain) -> tuple[BM25Okapi, list[dict[str, Any]]]:
        ### use of this function: build bm25 index
        """Build or retrieve cached BM25 index for a domain."""
        key = domain.value
        if key not in self._bm25_cache:
            chunks = self._store.get_all_chunks(domain)
            if not chunks:
                self._bm25_cache[key] = (BM25Okapi([[""]]), [])
                return self._bm25_cache[key]

            tokenized = [c["text"].lower().split() for c in chunks]
            bm25 = BM25Okapi(tokenized)
            self._bm25_cache[key] = (bm25, chunks)
        return self._bm25_cache[key]

    def _dense_search(self, domain: Domain, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        ### use of this function: dense search
        """Embed query and search ChromaDB."""
        model = _get_embed_model()
        query_embedding = model.encode(query).tolist()
        results = self._store.query_dense(domain, query_embedding, n_results=top_k)

        return [
            RetrievedChunk(
                chunk_text=r["text"],
                source_url=r.get("source_url", ""),
                source_title=r.get("source_title", ""),
                relevance_score=r.get("score", 0.0),
                domain=domain,
            )
            for r in results
        ]

    def _sparse_search(self, domain: Domain, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        ### use of this function: sparse search
        """BM25 sparse search."""
        bm25, chunks = self._build_bm25_index(domain)
        if not chunks:
            return []

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                chunk_text=chunks[idx]["text"],
                source_url=chunks[idx].get("source_url", ""),
                source_title=chunks[idx].get("source_title", ""),
                relevance_score=float(score),
                domain=domain,
            )
            for idx, score in scored_indices
            if score > 0
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        ### use of this function: reciprocal rank fusion
        """RRF: score = sum(1 / (k + rank)). Merge and deduplicate."""
        chunk_map: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense_results):
            key = chunk.chunk_text[:200] 
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

        for rank, chunk in enumerate(sparse_results):
            key = chunk.chunk_text[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [
            RetrievedChunk(
                chunk_text=chunk_map[key].chunk_text,
                source_url=chunk_map[key].source_url,
                source_title=chunk_map[key].source_title,
                relevance_score=rrf_scores[key],
                domain=chunk_map[key].domain,
            )
            for key in sorted_keys
        ]

    def _rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 3) -> list[RetrievedChunk]:
        ### use of this function: rerank
        """Cross-encoder reranking of candidate chunks."""
        if not candidates:
            return []

        cross_encoder = _get_cross_encoder()
        pairs = [(query, c.chunk_text) for c in candidates]
        scores = cross_encoder.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                chunk_text=chunk.chunk_text,
                source_url=chunk.source_url,
                source_title=chunk.source_title,
                relevance_score=float(score),
                domain=chunk.domain,
            )
            for chunk, score in scored[:top_k]
        ]

    def retrieve(self, domain: Domain, query: str, top_k: int = 6) -> RetrievalResult:
        ### use of this function: retrieve
        """
        Full hybrid retrieval pipeline:
        1. Dense search (top 20)
        2. BM25 sparse search (top 20)
        3. Reciprocal Rank Fusion
        4. Cross-encoder rerank → top_k
        """
        dense_results = self._dense_search(domain, query, top_k=20)
        sparse_results = self._sparse_search(domain, query, top_k=20)

        total_candidates = len(dense_results) + len(sparse_results)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        reranked = self._rerank(query, fused[:15], top_k=top_k)

        return RetrievalResult(
            chunks=reranked,
            query_used=query,
            total_candidates=total_candidates,
        )



_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    ### use of this function: get retriever
    """Get or create the singleton HybridRetriever."""
    global _retriever
    if _retriever is None:
        store = CorpusStore(settings.chroma_persist_dir)
        _retriever = HybridRetriever(store)
    return _retriever
