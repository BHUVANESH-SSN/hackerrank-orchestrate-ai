#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent — Interactive Terminal Mode.
"""
import sys
import time

from rich.console import Console

from config.settings import settings
from graph.pipeline import get_pipeline
from models.schemas import Domain
from rag.corpus_store import CorpusStore
from ui.terminal import print_banner, interactive_loop, print_corpus_stats
from utils.logger import ChatTranscriptLogger

console = Console()


def check_corpus_ready() -> bool:
    """Check if corpus has been built."""
    store = CorpusStore(settings.chroma_persist_dir)
    for domain in [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]:
        if not store.collection_exists(domain) or store.chunk_count(domain) == 0:
            return False
    return True


def main() -> None:
    print_banner()

    if not check_corpus_ready():
        console.print("[yellow]⚠️  Corpus not found. Run: python scripts/build_corpus.py[/yellow]")
        console.print("[dim]This ingests the support pages and builds the local vector database.[/dim]")
        sys.exit(1)

    # Show corpus stats
    store = CorpusStore(settings.chroma_persist_dir)
    stats = {d.value: store.chunk_count(d) for d in [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]}
    print_corpus_stats(stats)

    pipeline = get_pipeline()
    transcript = ChatTranscriptLogger()

    interactive_loop(pipeline, transcript)

    transcript.save(settings.log_file)
    console.print(f"\n[green]✅ Session log saved to {settings.log_file}[/green]")


if __name__ == "__main__":
    main()
