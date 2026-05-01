#!/usr/bin/env python3
"""
Batch processor: reads support_tickets.csv → writes output.csv + log.txt.
"""
import time

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from agents.output_formatter import format_csv_row
from config.settings import settings
from graph.pipeline import get_pipeline
from graph.state import GraphState
from models.schemas import TicketOutput
from utils.logger import ChatTranscriptLogger

console = Console()


def run_pipeline_on_ticket(
    pipeline, ticket_id: str, ticket_text: str, subject: str = "", company: str = ""
) -> TicketOutput:
    """Run the full pipeline on a single ticket."""
    state: GraphState = {
        "ticket_id": ticket_id,
        "raw_ticket": ticket_text,
        "subject": subject,
        "company": company,
        "intake": None,
        "domain_result": None,
        "risk_result": None,
        "retrieval_result": None,
        "synthesis_result": None,
        "faithfulness_result": None,
        "final_output": None,
        "error": None,
        "start_time_ms": int(time.time() * 1000),
    }
    result = pipeline.invoke(state)
    return result["final_output"]


def main() -> None:
    console.print("[bold cyan]🤖 Batch Processor — Support Triage Agent[/]\n")

    # Load input CSV
    input_path = "../support_tickets/support_tickets.csv"
    df = pd.read_csv(input_path)
    console.print(f"[dim]Loaded {len(df)} tickets from {input_path}[/]")
    console.print(f"[dim]Columns: {list(df.columns)}[/]\n")

    # Detect columns — the CSV has: Issue, Subject, Company
    text_col = None
    for col in ["Issue", "issue", "text", "description", "message", "content", "body", "ticket"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    subject_col = None
    for col in ["Subject", "subject"]:
        if col in df.columns:
            subject_col = col
            break

    company_col = None
    for col in ["Company", "company"]:
        if col in df.columns:
            company_col = col
            break

    pipeline = get_pipeline()
    transcript = ChatTranscriptLogger()
    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing tickets...", total=len(df))

        for idx, row in df.iterrows():
            ticket_id = str(idx + 1)
            ticket_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            subject = str(row[subject_col]) if subject_col and pd.notna(row.get(subject_col, "")) else ""
            company = str(row[company_col]) if company_col and pd.notna(row.get(company_col, "")) else ""

            # Clean "None" strings
            if company.strip().lower() == "none":
                company = ""

            transcript.log_ticket_start(ticket_id, ticket_text)

            try:
                output = run_pipeline_on_ticket(pipeline, ticket_id, ticket_text, subject, company)
                csv_row = format_csv_row(output)
                results.append(csv_row)

                transcript.log_agent_step(
                    "batch_processor",
                    f"Ticket {ticket_id}",
                    f"{output.action.value} | {output.domain.value} | {output.product_area.value}",
                    ticket_id,
                )
                transcript.log_final_output(
                    ticket_id,
                    output.action.value,
                    output.response,
                    output.sources,
                )

                progress.update(task, description=f"Ticket {ticket_id}: {output.action.value}")

            except Exception as e:
                console.print(f"[red]Error on ticket {ticket_id}: {e}[/]")
                results.append({
                    "status": "Escalated",
                    "product_area": "unknown",
                    "response": f"System error — escalating for manual review. Error: {str(e)[:100]}",
                    "justification": "Pipeline error",
                    "request_type": "product_issue",
                })
                transcript.log_agent_step(
                    "batch_processor",
                    f"Ticket {ticket_id}",
                    f"ERROR: {str(e)[:80]}",
                    ticket_id,
                )

            progress.advance(task)
            time.sleep(6)  # Prevent Groq API rate limits (30 RPM free tier). 6 seconds = 10 tickets/min = ~25 calls/min

    # Write output CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(settings.output_csv, index=False)
    console.print(f"\n[green]✅ Output written to {settings.output_csv} ({len(results)} tickets)[/]")

    # Save transcript to AGENTS.md mandated location + local copy
    from pathlib import Path
    home_log = Path.home() / "hackerrank_orchestrate" / "log.txt"
    home_log.parent.mkdir(parents=True, exist_ok=True)
    transcript.save(str(home_log))
    transcript.save(settings.log_file)
    console.print(f"[green]✅ Log saved to {home_log}[/]")
    console.print(f"[green]✅ Log copy saved to {settings.log_file}[/]")


if __name__ == "__main__":
    main()
