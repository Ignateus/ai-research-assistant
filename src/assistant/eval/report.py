"""Eval report — format and save harness results."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .harness import HarnessResult
from .metrics import ALL_METRICS

console = Console()


def print_summary(result: HarnessResult) -> None:
    """Render a Rich table summarising per-metric scores."""
    summary = result.summary()

    table = Table(
        title=f"Eval Results — {result.dataset_name}",
        title_style="bold cyan",
        border_style="cyan",
        show_footer=True,
    )

    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="center")
    table.add_column("Std Dev", justify="center")
    table.add_column("Bar", justify="left", min_width=20)

    for metric in ALL_METRICS:
        stats = summary["metrics"][metric.value]
        m = stats["mean"]
        s = stats["std"]

        # Simple ASCII bar (each █ = 1 point out of 5)
        filled = round(m)
        bar = "█" * filled + "░" * (5 - filled)
        colour = "green" if m >= 4 else ("yellow" if m >= 3 else "red")

        table.add_row(
            metric.value,
            f"{m:.2f}",
            f"{s:.2f}",
            Text(bar, style=colour),
        )

    console.print()
    console.print(table)
    console.print(
        f"  Overall mean: [bold]{summary['mean_overall']:.2f}[/bold] / 5.00  "
        f"  ({summary['n_questions']} questions evaluated)"
    )
    console.print()


def print_detail(result: HarnessResult) -> None:
    """Print per-question scores with reasoning."""
    for es in result.eval_scores:
        console.rule(f"[dim]{es.question_id}[/dim]", style="dim")
        console.print(f"[bold]Q:[/bold] {es.question}")
        console.print(f"[bold]A:[/bold] [dim]{es.response[:200]}{'…' if len(es.response) > 200 else ''}[/dim]")
        console.print()
        for score in es.scores:
            colour = "green" if score.score >= 4 else ("yellow" if score.score >= 3 else "red")
            console.print(
                f"  [{colour}]{score.score}/5[/{colour}]  "
                f"[bold]{score.metric.value}[/bold]  —  [dim]{score.reasoning}[/dim]"
            )
        if es.error:
            console.print(f"  [red]Error:[/red] {es.error}")
        console.print()


def save_report(result: HarnessResult, path: Path) -> None:
    """Save the full eval result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))
    console.print(f"[dim]Report saved → {path}[/dim]")
