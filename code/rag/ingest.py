"""
Corpus ingestion pipeline — reads pre-scraped markdown files from data/ directory,
chunks them, embeds with sentence-transformers, and stores in ChromaDB.
"""
import uuid
from pathlib import Path
from typing import Any

import tiktoken
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from sentence_transformers import SentenceTransformer

from config.settings import settings
from models.schemas import Domain
from rag.corpus_store import CorpusStore

# ──────────────────────────────────────────────
# Domain → subdirectory mapping
# ──────────────────────────────────────────────

DOMAIN_DIRS: dict[Domain, str] = {
    Domain.HACKERRANK: "hackerrank",
    Domain.CLAUDE: "claude",
    Domain.VISA: "visa",
}

# Singleton embedding model
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ──────────────────────────────────────────────
# Read local corpus files
# ──────────────────────────────────────────────

def read_corpus_files(corpus_dir: str, domain: Domain) -> list[dict[str, str]]:
    """
    Read all .md files from the domain's subdirectory in corpus_dir.
    Returns list of {url, title, text} dicts.
    """
    domain_dir = Path(corpus_dir) / DOMAIN_DIRS[domain]
    if not domain_dir.exists():
        return []

    documents: list[dict[str, str]] = []
    for md_file in sorted(domain_dir.rglob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text or len(text) < 50:
                continue

            # Derive title from first heading or filename
            title = md_file.stem.replace("-", " ").replace("_", " ").title()
            lines = text.split("\n")
            for line in lines[:5]:
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break

            # Build a pseudo URL from file path
            rel_path = md_file.relative_to(domain_dir)
            match domain:
                case Domain.HACKERRANK:
                    source_url = f"https://support.hackerrank.com/hc/en-us/{rel_path}"
                case Domain.CLAUDE:
                    source_url = f"https://support.claude.com/en/{rel_path}"
                case Domain.VISA:
                    source_url = f"https://www.visa.co.in/support/{rel_path}"
                case _:
                    source_url = str(md_file)

            documents.append({
                "url": source_url,
                "title": title,
                "text": text,
            })
        except Exception:
            continue

    return documents


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    source_url: str,
    source_title: str,
    domain: Domain,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Sliding-window chunking based on token count (tiktoken cl100k_base).
    Returns list of chunk dicts with id, text, source_url, source_title, domain.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) == 0:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens).strip()

        if len(chunk_text_str) > 20:  # Skip trivially small chunks
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text_str,
                "source_url": source_url,
                "source_title": source_title,
                "domain": domain.value,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
        start += step

    return chunks


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Embed all chunk texts using all-MiniLM-L6-v2. Adds 'embedding' field."""
    if not chunks:
        return chunks

    model = _get_embed_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


# ──────────────────────────────────────────────
# Domain Ingestion
# ──────────────────────────────────────────────

def ingest_domain(domain: Domain, store: CorpusStore) -> int:
    """
    Full ingestion pipeline for one domain: read files → chunk → embed → store.
    Returns number of chunks stored. Idempotent — skips if collection already has data.
    """
    if store.collection_exists(domain):
        existing = store.chunk_count(domain)
        if existing > 0:
            return existing

    documents = read_corpus_files(settings.corpus_dir, domain)
    if not documents:
        return 0

    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            source_url=doc["url"],
            source_title=doc["title"],
            domain=domain,
            chunk_size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # Embed
    all_chunks = embed_chunks(all_chunks)

    # Store
    store.add_chunks(domain, all_chunks)

    return len(all_chunks)


# ──────────────────────────────────────────────
# Full Corpus Build
# ──────────────────────────────────────────────

def build_corpus() -> dict[str, int]:
    """Run ingestion for all three domains. Returns {domain: chunk_count}."""
    store = CorpusStore(settings.chroma_persist_dir)
    results: dict[str, int] = {}

    domains = [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Building corpus...", total=len(domains))

        for domain in domains:
            progress.update(task, description=f"Ingesting {domain.value}...")
            count = ingest_domain(domain, store)
            results[domain.value] = count
            progress.advance(task)

    return results
