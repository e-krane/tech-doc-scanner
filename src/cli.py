"""
Command-line interface for the document conversion agent.
"""

import click
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import AppConfig, ConverterConfig
from .converter import TechDocConverter
from .utils import setup_logging, ConversionStats


console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output except errors')
@click.option('--log-file', type=click.Path(), help='Log to file')
@click.pass_context
def cli(ctx, verbose, quiet, log_file):
    """Technical Document Conversion Agent powered by Docling."""
    ctx.ensure_object(dict)

    # Setup logging
    log_path = Path(log_file) if log_file else None
    setup_logging(verbose=verbose, quiet=quiet, log_file=log_path)

    # Store flags in context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='output',
              help='Output directory for converted files')
@click.option('--no-figures', is_flag=True, help='Disable figure extraction')
@click.option('--profile', is_flag=True, help='Enable performance profiling')
@click.pass_context
def convert(ctx, file_path, output_dir, no_figures, profile):
    """Convert a single document to markdown."""
    file_path = Path(file_path)

    # Create converter config
    config = ConverterConfig(
        output_dir=Path(output_dir),
        save_figures=not no_figures,
        enable_profiling=profile
    )

    # Create converter
    converter = TechDocConverter(config)

    # Convert document
    console.print(f"\n[bold blue]Converting:[/bold blue] {file_path.name}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=None)

        result = converter.convert_and_save(file_path)

        progress.update(task, completed=True)

    if result.success:
        console.print(f"[green]✓[/green] Conversion successful!")
        console.print(f"  Pages: {result.page_count}")
        console.print(f"  Time: {result.conversion_time:.2f}s")
        if result.figure_count > 0:
            console.print(f"  Figures: {result.figure_count}")
            console.print(f"  Figures dir: {result.figures_dir}")
    else:
        console.print(f"[red]✗[/red] Conversion failed: {result.error}")
        return 1

    return 0


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='output',
              help='Output directory for converted files')
@click.option('--no-figures', is_flag=True, help='Disable figure extraction')
@click.option('--pattern', '-p', default='*.pdf', help='File pattern to match')
@click.pass_context
def batch(ctx, input_dir, output_dir, no_figures, pattern):
    """Convert all documents in a directory."""
    input_dir = Path(input_dir)

    # Find files to convert
    files = list(input_dir.rglob(pattern))

    if not files:
        console.print(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
        return 0

    console.print(f"\n[bold blue]Found {len(files)} file(s) to convert[/bold blue]\n")

    # Create converter config
    config = ConverterConfig(
        output_dir=Path(output_dir),
        save_figures=not no_figures
    )

    # Create converter and stats
    converter = TechDocConverter(config)
    stats = ConversionStats()

    # Process files with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Converting documents...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Processing {file_path.name}...")

            result = converter.convert_and_save(file_path)

            if result.success:
                stats.add_success(
                    pages=result.page_count,
                    figures=result.figure_count,
                    time=result.conversion_time
                )
            else:
                stats.add_failure(file_path, result.error)

            progress.advance(task)

    # Print summary
    console.print(stats.get_report())

    return 0 if stats.failed == 0 else 1


if __name__ == '__main__':
    cli(obj={})
