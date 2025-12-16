"""Pipeline CLI commands."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Pipeline execution commands")
console = Console()


@app.command("run")
def run_pipeline(
    pipeline_name: str = typer.Argument(..., help="Pipeline name"),
    input_path: Path = typer.Argument(..., help="Input path"),
    output_dir: Path = typer.Option("./output", "--output", "-o"),
) -> None:
    """Run a registered pipeline."""
    console.print(f"Running pipeline: {pipeline_name}")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_dir}")


@app.command("list")
def list_pipelines() -> None:
    """List available pipelines."""
    from rhenium.core.registry import get_registry
    pipelines = get_registry().list_pipelines()
    for p in pipelines:
        console.print(f"  - {p['name']} (v{p.get('version', '?')})")
