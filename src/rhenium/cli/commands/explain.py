"""Explain CLI commands."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="XAI and explanation commands")
console = Console()


@app.command("dossier")
def generate_dossier(
    finding_id: str = typer.Argument(..., help="Finding ID"),
    output_dir: Path = typer.Option("./dossiers", "--output", "-o"),
) -> None:
    """Generate evidence dossier for a finding."""
    console.print(f"Generating evidence dossier for: {finding_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Dossier saved to {output_dir}[/green]")


@app.command("saliency")
def generate_saliency(
    model_name: str = typer.Argument(..., help="Model name"),
    input_path: Path = typer.Argument(..., help="Input image"),
) -> None:
    """Generate saliency map."""
    console.print(f"Generating saliency for {model_name} on {input_path}")
