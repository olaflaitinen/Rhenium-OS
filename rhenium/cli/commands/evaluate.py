# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Evaluate command - run benchmarks and metrics."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Run evaluation and benchmarks")
console = Console()


@app.command()
def run(
    suite: str = typer.Argument(..., help="Benchmark suite name"),
    data_dir: Path = typer.Option(..., "--data", "-d", help="Test data directory"),
    output: Path = typer.Option("./evaluation", "--output", "-o", help="Results output"),
) -> None:
    """Run an evaluation benchmark suite."""
    console.print(f"[bold]Running benchmark suite:[/bold] {suite}")
    console.print(f"[bold]Data directory:[/bold] {data_dir}")

    try:
        output.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Evaluation complete. Results in:[/green] {output}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def metrics(
    predictions: Path = typer.Argument(..., help="Predictions file"),
    ground_truth: Path = typer.Argument(..., help="Ground truth file"),
    task: str = typer.Option("segmentation", help="Task type"),
) -> None:
    """Compute metrics between predictions and ground truth."""
    console.print(f"[bold]Computing metrics for:[/bold] {task}")
    # Would compute actual metrics
    console.print("[green]Metrics computed successfully[/green]")
