# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Rhenium OS CLI Main Entry Point
================================

Typer-based command-line interface.
"""

from __future__ import annotations

import typer
from rich.console import Console

from rhenium import __version__
from rhenium.cli.commands import ingest, run_pipeline, evaluate, explain, inspect

console = Console()

app = typer.Typer(
    name="rhenium",
    help="Skolyn Rhenium OS - AI Operating System for Diagnostic Imaging",
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(ingest.app, name="ingest")
app.add_typer(run_pipeline.app, name="run-pipeline")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(explain.app, name="explain")
app.add_typer(inspect.app, name="inspect")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """Skolyn Rhenium OS CLI."""
    if version:
        console.print(f"Rhenium OS version {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
