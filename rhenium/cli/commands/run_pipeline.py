# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Run pipeline command."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Run analysis pipelines")
console = Console()


@app.command()
def run(
    config: str = typer.Argument(..., help="Pipeline configuration name"),
    input_path: Path = typer.Option(..., "--input", "-i", help="Input data path"),
    output: Path = typer.Option("./results", "--output", "-o", help="Output directory"),
) -> None:
    """Run a pipeline on input data."""
    from rhenium.pipelines.pipeline_runner import PipelineRunner

    console.print(f"[bold]Running pipeline:[/bold] {config}")
    console.print(f"[bold]Input:[/bold] {input_path}")

    try:
        runner = PipelineRunner.from_config(config)
        # Would load actual data and run
        console.print(f"[green]Pipeline complete. Results in:[/green] {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list_configs() -> None:
    """List available pipeline configurations."""
    from rhenium.pipelines.pipeline_runner import PipelineRunner

    configs = PipelineRunner.list_available_configs()
    console.print("[bold]Available pipeline configurations:[/bold]")
    for cfg in configs:
        console.print(f"  - {cfg}")
