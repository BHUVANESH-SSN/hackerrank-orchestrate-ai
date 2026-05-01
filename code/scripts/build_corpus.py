#!/usr/bin/env python3
"""
One-time corpus ingestion runner.
Reads pre-scraped data from data/ → chunks → embeds → stores in ChromaDB.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

# Ensure code/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from models.schemas import Domain
from rag.corpus_store import CorpusStore
from rag.ingest import build_corpus, ingest_domain

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build support corpus")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion even if data exists")
    args = parser.parse_args()

    console.print("[bold cyan]📚 Support Corpus Builder[/]")
    console.print(f"[dim]Corpus source: {settings.corpus_dir}[/]")
    console.print(f"[dim]ChromaDB dir: {settings.chroma_persist_dir}[/]")
    console.print()

    store = CorpusStore(settings.chroma_persist_dir)

    # Check if already built
    domains = [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]
    all_exist = all(store.collection_exists(d) and store.chunk_count(d) > 0 for d in domains)

    if all_exist and not args.force:
        console.print("[green]✅ Corpus already built! Showing stats:[/]\n")
        _print_stats(store, domains)
        return

    if args.force:
        console.print("[yellow]⚠️  Force mode — re-ingesting all domains[/]\n")

    # Build corpus
    results = build_corpus()

    # Print results table
    table = Table(title="Ingestion Results", show_header=True)
    table.add_column("Domain", style="bold cyan")
    table.add_column("Chunks Stored", justify="right", style="green")

    for domain, count in results.items():
        table.add_row(domain, str(count))

    total = sum(results.values())
    table.add_row("[bold]Total[/]", f"[bold]{total}[/]")
    console.print(table)

    # Write manifest
    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": settings.corpus_dir,
        "domains": results,
        "total_chunks": total,
    }
    manifest_path = Path(settings.chroma_persist_dir).parent / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[green]✅ Corpus built successfully! Manifest: {manifest_path}[/]")


def _print_stats(store: CorpusStore, domains: list[Domain]) -> None:
    """Print current corpus statistics."""
    table = Table(title="Corpus Statistics", show_header=True)
    table.add_column("Domain", style="bold cyan")
    table.add_column("Chunks", justify="right", style="green")

    total = 0
    for d in domains:
        count = store.chunk_count(d)
        table.add_row(d.value, str(count))
        total += count

    table.add_row("[bold]Total[/]", f"[bold]{total}[/]")
    console.print(table)


if __name__ == "__main__":
    main()
