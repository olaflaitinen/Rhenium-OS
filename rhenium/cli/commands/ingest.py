# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Ingest command - DICOM and raw data ingestion."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Ingest DICOM or raw data into Rhenium OS")
console = Console()


@app.command()
def dicom(
    source: Path = typer.Argument(..., help="Path to DICOM directory"),
    output: Path = typer.Option("./data", help="Output directory"),
    recursive: bool = typer.Option(True, help="Search recursively"),
) -> None:
    """Ingest DICOM data from a directory."""
    from rhenium.data.dicom_io import load_dicom_study

    console.print(f"[bold]Ingesting DICOM from[/bold]: {source}")

    try:
        study = load_dicom_study(source, recursive=recursive)
        console.print(f"[green]Loaded study with {len(study.series)} series[/green]")

        # Store metadata
        output.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Data indexed to[/green]: {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def raw(
    source: Path = typer.Argument(..., help="Path to raw data file"),
    data_type: str = typer.Option("kspace", help="Data type: kspace, sinogram"),
    output: Path = typer.Option("./data", help="Output directory"),
) -> None:
    """Ingest raw acquisition data."""
    console.print(f"[bold]Ingesting {data_type} from[/bold]: {source}")

    try:
        if data_type == "kspace":
            from rhenium.data.raw_io import load_kspace
            data = load_kspace(source)
            console.print(f"[green]Loaded k-space: shape={data.shape}[/green]")
        elif data_type == "sinogram":
            from rhenium.data.raw_io import load_sinogram
            data = load_sinogram(source)
            console.print(f"[green]Loaded sinogram: shape={data.shape}[/green]")
        else:
            console.print(f"[red]Unknown data type: {data_type}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
