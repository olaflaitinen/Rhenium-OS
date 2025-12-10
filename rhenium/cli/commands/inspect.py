# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Inspect command - inspect registry and models."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Inspect registry, models, and configurations")
console = Console()


@app.command()
def registry() -> None:
    """List registered components."""
    from rhenium.core.registry import registry as reg, ComponentType

    table = Table(title="Registered Components")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version")

    for ctype in ComponentType:
        components = reg.list_components(component_type=ctype)
        for comp in components:
            table.add_row(ctype.value, comp.name, comp.version)

    console.print(table)


@app.command()
def config() -> None:
    """Show current configuration."""
    from rhenium.core.config import get_settings

    settings = get_settings()
    console.print("[bold]Current Configuration:[/bold]")
    console.print(f"  Data directory: {settings.data_dir}")
    console.print(f"  Models directory: {settings.models_dir}")
    console.print(f"  Device: {settings.device.value}")
    console.print(f"  MedGemma backend: {settings.medgemma_backend.value}")


@app.command()
def version() -> None:
    """Show version information."""
    from rhenium import __version__
    console.print(f"Rhenium OS version: {__version__}")
