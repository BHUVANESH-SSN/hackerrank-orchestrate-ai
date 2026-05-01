"""
ChromaDB wrapper — one persistent collection per domain.
"""
from pathlib import Path
from typing import Any

import chromadb

from models.schemas import Domain


class CorpusStore:
    """Manages ChromaDB collections for the three support domains."""

    def __init__(self, persist_dir: str) -> None:
        ### use of this function: init
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict[str, chromadb.Collection] = {}

    def _collection_name(self, domain: Domain) -> str:
        ### use of this function: collection name
        return f"support_{domain.value}"

    def get_or_create_collection(self, domain: Domain) -> chromadb.Collection:
        ### use of this function: get or create collection
        """Get or create a ChromaDB collection for the given domain."""
        name = self._collection_name(domain)
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def add_chunks(self, domain: Domain, chunks: list[dict[str, Any]]) -> None:
        ### use of this function: add chunks
        """Add chunks to the domain's collection. Each chunk has: id, text, embedding, source_url, source_title, domain."""
        collection = self.get_or_create_collection(domain)

        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=[c["embedding"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "source_url": c.get("source_url", ""),
                        "source_title": c.get("source_title", ""),
                        "domain": c.get("domain", domain.value),
                        "chunk_index": c.get("chunk_index", 0),
                    }
                    for c in batch
                ],
            )

    def query_dense(
        self, domain: Domain, query_embedding: list[float], n_results: int = 20
    ) -> list[dict[str, Any]]:
        ### use of this function: query dense
        """Query the domain collection with a dense embedding vector."""
        collection = self.get_or_create_collection(domain)
        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            for idx in range(len(results["documents"][0])):
                output.append(
                    {
                        "text": results["documents"][0][idx],
                        "source_url": results["metadatas"][0][idx].get("source_url", ""),
                        "source_title": results["metadatas"][0][idx].get("source_title", ""),
                        "domain": results["metadatas"][0][idx].get("domain", domain.value),
                        "score": 1.0 - results["distances"][0][idx], 
                    }
                )
        return output

    def collection_exists(self, domain: Domain) -> bool:
        ### use of this function: collection exists
        """Check if a collection exists and has data."""
        name = self._collection_name(domain)
        try:
            col = self._client.get_collection(name)
            return col.count() > 0
        except Exception:
            return False

    def get_all_chunks(self, domain: Domain) -> list[dict[str, Any]]:
        ### use of this function: get all chunks
        """Retrieve all chunks from a domain collection."""
        collection = self.get_or_create_collection(domain)
        count = collection.count()
        if count == 0:
            return []

        results = collection.get(include=["documents", "metadatas"])
        output: list[dict[str, Any]] = []
        for idx in range(len(results["documents"])):
            output.append(
                {
                    "id": results["ids"][idx],
                    "text": results["documents"][idx],
                    "source_url": results["metadatas"][idx].get("source_url", ""),
                    "source_title": results["metadatas"][idx].get("source_title", ""),
                    "domain": results["metadatas"][idx].get("domain", domain.value),
                }
            )
        return output

    def chunk_count(self, domain: Domain) -> int:
        ### use of this function: chunk count
        """Return the number of chunks in a domain collection."""
        try:
            collection = self.get_or_create_collection(domain)
            return collection.count()
        except Exception:
            return 0
