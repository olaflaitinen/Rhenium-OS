"""Rhenium OS Command Line Interface."""

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
import rhenium

app = typer.Typer(
    name="rhenium",
    help="Rhenium OS - Multi-Modality AI Platform for Medical Imaging Research",
    add_completion=True,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold]Rhenium OS[/bold] v{rhenium.__version__}")


@app.command()
def info() -> None:
    """Show system information."""
    from rhenium.core.config import get_settings
    settings = get_settings()
    console.print(f"[bold]Data dir:[/bold] {settings.data_dir}")
    console.print(f"[bold]Models dir:[/bold] {settings.models_dir}")
    console.print(f"[bold]Device:[/bold] {settings.get_effective_device()}")


@app.command()
def ingest(
    input_path: Path = typer.Argument(..., help="Input DICOM directory or NIfTI file"),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Ingest medical imaging data."""
    from rhenium.data.dicom import load_dicom_directory
    from rhenium.data.nifti import load_nifti

    if input_path.is_dir():
        console.print(f"Loading DICOM from {input_path}...")
        study = load_dicom_directory(input_path)
        console.print(f"Loaded {len(study.series)} series")
    elif input_path.suffix in [".nii", ".gz"]:
        console.print(f"Loading NIfTI from {input_path}...")
        volume = load_nifti(input_path)
        console.print(f"Loaded volume: {volume.shape}")
    else:
        console.print("[red]Unknown file type[/red]")


@app.command()
def synthetic(
    output_dir: Path = typer.Option("./data/synthetic", "--output", "-o"),
    num_studies: int = typer.Option(3, "--num", "-n"),
) -> None:
    """Generate synthetic test data."""
    from rhenium.testing.synthetic import SyntheticDataGenerator
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    gen = SyntheticDataGenerator(seed=42)

    for i in range(num_studies):
        study = gen.generate_study(num_series=2)
        study_dir = output_dir / f"study_{i+1:03d}"
        study_dir.mkdir(exist_ok=True)
        with open(study_dir / "metadata.json", "w") as f:
            json.dump(study.to_dict(), f, indent=2)
        console.print(f"Generated {study_dir}")

    console.print(f"[green]Generated {num_studies} synthetic studies[/green]")


@app.command()
def benchmark(
    output_dir: Path = typer.Option("./results", "--output", "-o"),
) -> None:
    """Run benchmark on synthetic data."""
    import numpy as np
    from rhenium.testing.synthetic import SyntheticDataGenerator
    from rhenium.evaluation import dice_score, psnr

    gen = SyntheticDataGenerator(seed=42)
    volume = gen.generate_volume(shape=(32, 64, 64), add_lesion=True)
    mask = gen.generate_segmentation_mask(shape=(32, 64, 64), num_classes=2)

    pred = (mask > 0).astype(np.float32)
    target = (mask > 0).astype(np.float32)

    dice = dice_score(pred, target)
    psnr_val = psnr(volume.array, volume.array)

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Dice Score", f"{dice:.4f}")
    table.add_row("PSNR (dB)", f"{psnr_val:.2f}")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn
    console.print(f"Starting server at http://{host}:{port}")
    uvicorn.run("rhenium.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
