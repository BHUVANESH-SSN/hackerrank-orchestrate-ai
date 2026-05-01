"""
Rich terminal interface for the support triage agent.
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

from models.schemas import Action, TicketOutput


console = Console()


def print_banner() -> None:
    ### use of this function: print banner
    """Print a styled banner."""
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║    🤖 Multi-Domain Support Triage Agent         ║\n", style="bold cyan")
    banner.append("║    HackerRank Orchestrate Challenge              ║\n", style="bold cyan")
    banner.append("╠══════════════════════════════════════════════════╣\n", style="bold cyan")
    banner.append("║  Domains:                                        ║\n", style="cyan")
    banner.append("║  • HackerRank  → support.hackerrank.com          ║\n", style="cyan")
    banner.append("║  • Claude      → support.claude.com              ║\n", style="cyan")
    banner.append("║  • Visa        → visa.co.in/support              ║\n", style="cyan")
    banner.append("╚══════════════════════════════════════════════════╝\n", style="bold cyan")
    console.print(banner)


def print_ticket_result(output: TicketOutput) -> None:
    ### use of this function: print ticket result
    """Print a rich panel with ticket processing results."""
    if output.action == Action.ESCALATE:
        border_color = "red"
        action_style = "[bold red]ESCALATED[/]"
    elif output.action == Action.PARTIAL_REPLY:
        border_color = "yellow"
        action_style = "[bold yellow]PARTIAL REPLY[/]"
    else:
        border_color = "green"
        action_style = "[bold green]REPLIED[/]"

    lines: list[str] = []
    lines.append(f"[bold]Domain:[/] {output.domain.value} | [bold]Product Area:[/] {output.product_area.value}")
    lines.append(f"[bold]Request Type:[/] {output.request_type.value}")
    lines.append(f"[bold]Risk Score:[/] {output.risk_score}/10")

    if output.escalation_reason:
        lines.append(f"[bold yellow]Escalation:[/] {output.escalation_reason}")
        if output.escalation_team:
            lines.append(f"[bold yellow]Team:[/] {output.escalation_team}")

    lines.append("")
    lines.append("[bold underline]Response:[/]")
    lines.append(output.response)

    if output.sources:
        lines.append("")
        lines.append("[bold underline]Sources:[/]")
        for src in output.sources:
            lines.append(f"  • {src}")

    if output.faithfulness_score is not None:
        lines.append(f"\n[dim]Faithfulness: {output.faithfulness_score:.2f}[/]")

    lines.append(f"[dim]Processing time: {output.processing_time_ms}ms[/]")

    content = "\n".join(lines)
    title = f" Ticket {output.ticket_id} | {output.domain.value} | {action_style} "

    console.print(Panel(content, title=title, border_style=border_color, expand=True))


def print_processing_step(step_name: str, status: str) -> None:
    ### use of this function: print processing step
    """Print a processing step indicator."""
    console.print(f"  [dim]→[/] [bold]{step_name}[/]: {status}")


def print_corpus_stats(stats: dict[str, int]) -> None:
    ### use of this function: print corpus stats
    """Print a table of domain → chunk count."""
    table = Table(title="📚 Corpus Statistics", show_header=True)
    table.add_column("Domain", style="bold cyan")
    table.add_column("Chunks", justify="right", style="green")

    for domain, count in stats.items():
        table.add_row(domain, str(count))

    total = sum(stats.values())
    table.add_row("[bold]Total[/]", f"[bold]{total}[/]")

    console.print(table)
    console.print()


def interactive_loop(pipeline, transcript=None) -> None:
    ### use of this function: interactive loop
    """Main REPL loop for interactive ticket processing."""
    import time
    from graph.state import GraphState

    console.print("[bold green]Ready![/] Enter a support ticket (or 'quit' to exit).\n")

    ticket_num = 0
    while True:
        try:
            ticket_text = Prompt.ask("\n[bold cyan]📝 Ticket[/]")
        except (KeyboardInterrupt, EOFError):
            break

        if ticket_text.strip().lower() in ("exit", "quit", "q", ""):
            break

        ticket_num += 1
        ticket_id = f"interactive_{ticket_num}"

        if transcript:
            transcript.log_ticket_start(ticket_id, ticket_text)

        console.print(f"\n[dim]Processing ticket {ticket_id}...[/]")

        state: GraphState = {
            "ticket_id": ticket_id,
            "raw_ticket": ticket_text,
            "subject": "",
            "company": "",
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

        try:
            result = pipeline.invoke(state)
            output = result.get("final_output")
            if output:
                print_ticket_result(output)
                if transcript:
                    transcript.log_agent_step(
                        "pipeline",
                        ticket_text[:100],
                        f"{output.action.value} | {output.domain.value}",
                        ticket_id,
                    )
                    transcript.log_final_output(
                        ticket_id,
                        output.action.value,
                        output.response,
                        output.sources,
                    )
            else:
                console.print("[red]Error: No output generated[/]")
        except Exception as e:
            console.print(f"[red]Pipeline error: {e}[/]")

    console.print("\n[bold cyan]👋 Goodbye![/]")
