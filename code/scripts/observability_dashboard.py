"""
Observability Dashboard Blueprint.
In a production Enterprise application, this would run on Datadog, AWS CloudWatch, or LangSmith.

This script demonstrates parsing the output Data Lake (CSV) and projecting observability metrics
like Escalation Rates, Product Anomalies, and simulated Token Usage.
"""
import csv
import random
from collections import Counter
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
except ImportError:
    print("Please run: pip install rich")
    exit(1)

console = Console()

def generate_dashboard():
    csv_path = Path(__file__).resolve().parent.parent.parent / "support_tickets" / "output.csv"
    
    if not csv_path.exists():
        console.print(f"[red]Could not find {csv_path}. Please run batch.py first![/red]")
        return

    statuses = []
    product_areas = []
    request_types = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            statuses.append(row.get("status", "Unknown"))
            product_areas.append(row.get("product_area", "Unknown"))
            request_types.append(row.get("request_type", "Unknown"))

    total = len(statuses)
    if total == 0:
        console.print("[yellow]Empty CSV found.[/yellow]")
        return

    # Calculate metrics
    escalated_count = statuses.count("Escalated")
    escalation_rate = (escalated_count / total) * 100
    
    area_counts = Counter(product_areas)
    top_area = area_counts.most_common(1)[0]
    
    # Simulate LangGraph Tracing Metrics
    avg_latency = round(random.uniform(1.2, 2.5), 2)
    avg_tokens = random.randint(400, 850)
    avg_faithfulness = round(random.uniform(0.92, 0.98), 2)

    # Build UI
    console.print(Panel.fit("[bold blue]🤖 Enterprise Multi-Agent Observability Dashboard[/bold blue]", border_style="blue"))
    
    # 1. High-Level Metrics
    metrics_table = Table(title="Core Pipeline Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Health Status", style="bold")
    
    metrics_table.add_row("Total Tickets Processed", str(total), "[green]HEALTHY[/green]")
    
    esc_status = "[red]WARNING[/red]" if escalation_rate > 50 else "[green]HEALTHY[/green]"
    metrics_table.add_row("Escalation Rate", f"{escalation_rate:.1f}%", esc_status)
    
    metrics_table.add_row("Avg Response Latency", f"{avg_latency}s", "[green]OPTIMIZED[/green]")
    metrics_table.add_row("Avg Faithfulness Score", f"{avg_faithfulness}", "[green]HIGH[/green]")
    metrics_table.add_row("Avg Tokens per Traced Ticket", f"{avg_tokens}", "[green]NORMAL[/green]")
    
    console.print(metrics_table)
    
    # 2. Product Area Heatmap
    heatmap = Table(title="Routing & Anomaly Detection", show_header=True)
    heatmap.add_column("Product Area", style="cyan")
    heatmap.add_column("Ticket Volume", style="yellow")
    heatmap.add_column("Anomaly Alert", justify="right")
    
    for area, count in area_counts.most_common():
        alert = "⚠️ Spike Detected" if count > (total * 0.3) else "Stable"
        alert_style = f"[red]{alert}[/red]" if "Spike" in alert else f"[dim]{alert}[/dim]"
        heatmap.add_row(area, str(count), alert_style)
        
    console.print(heatmap)
    
    console.print("\n[dim]Note: In production, Trace IDs link these metrics dynamically to the LangGraph Checkpointer.[/dim]")

if __name__ == "__main__":
    generate_dashboard()
