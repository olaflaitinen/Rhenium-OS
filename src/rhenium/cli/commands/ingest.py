"""Ingest CLI commands."""

from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer(help="Data ingestion commands")
console = Console()


@app.command("dicom")
def ingest_dicom(
    input_path: Path = typer.Argument(..., help="DICOM directory"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
    deidentify: bool = typer.Option(True, "--deidentify/--no-deidentify"),
) -> None:
    """Ingest DICOM data."""
    from rhenium.data.dicom import load_dicom_directory

    console.print(f"Loading DICOM from {input_path}...")
    study = load_dicom_directory(input_path, deidentify=deidentify)
    console.print(f"[green]Loaded {len(study.series)} series[/green]")


@app.command("nifti")
def ingest_nifti(
    input_path: Path = typer.Argument(..., help="NIfTI file"),
) -> None:
    """Ingest NIfTI data."""
    from rhenium.data.nifti import load_nifti

    console.print(f"Loading NIfTI from {input_path}...")
    volume = load_nifti(input_path)
    console.print(f"[green]Loaded volume: {volume.shape}[/green]")
