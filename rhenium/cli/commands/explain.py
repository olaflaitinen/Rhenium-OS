# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Explain command - generate XAI artifacts."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Generate explanations and evidence dossiers")
console = Console()


@app.command()
def generate(
    results: Path = typer.Argument(..., help="Path to pipeline results"),
    output: Path = typer.Option("./explanations", help="Output directory"),
) -> None:
    """Generate explanations for pipeline results."""
    console.print(f"[bold]Generating explanations for:[/bold] {results}")

    try:
        output.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Explanations saved to:[/green] {output}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def dossier(
    finding_id: str = typer.Argument(..., help="Finding ID"),
    results_dir: Path = typer.Option("./results", help="Results directory"),
) -> None:
    """Display evidence dossier for a finding."""
    console.print(f"[bold]Evidence dossier for finding:[/bold] {finding_id}")
    # Would load and display dossier
